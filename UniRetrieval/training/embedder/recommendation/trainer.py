from collections import defaultdict
import json
import os
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

from UniRetrieval.training.embedder.recommendation.arguments import get_logger
# TODO eval部分暂未移植
# from rs4industry.eval import get_eval_metrics
from .arguments import TrainingArguments
from .datasets import Callback, EarlyStopCallback, CheckpointCallback
from UniRetrieval.abc.training.embedder import AbsEmbedderTrainer

import sys
from typing import *

import torch.nn.functional as F
import torchmetrics.functional as M

# copied from rec studio Trainer
# TODO 添加datacollator逻辑?
class RetrieverTrainer(AbsEmbedderTrainer):
    def __init__(self, model, config=None, train=True, *args, **kwargs):
        super(RetrieverTrainer, self).__init__(model, *args, **kwargs)
        self.config: TrainingArguments = self.load_config(config)
        self.train_mode = train
        self.model_type = model.model_type
        self._check_checkpoint_dir()

        self.accelerator = Accelerator()
        if self.accelerator.is_main_process:
            print(model)
        model.to(self.accelerator.device)
        optimizer = self.get_optimizer(
            self.config.optimizer,
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay    
        )
        self.model = self.accelerator.prepare(model)
        self.optimizer = self.accelerator.prepare(optimizer)
        self.lr_scheduler = self.get_lr_scheduler()
        self.logger = get_logger(self.config)
        self.global_step = 0    # global step counter, load from checkpoint if exists
        self.cur_global_step = 0    # current global step counter, record the steps of this training
        self._last_eval_epoch = -1
        if self.train_mode:
            self.callbacks: List[Callback] = self.get_callbacks()
        self._total_train_samples = 0
        self._total_eval_samples = 0
        self.item_vectors = None
        self.item_ids = None
        
    def compute_loss(self, model, batch, return_outputs=False,*args, **kwargs):
        outputs = model(batch=batch, cal_loss=True,*args, **kwargs)
        loss = outputs['loss']

        return (loss, outputs) if return_outputs else loss
    
    # TODO
    def save_model(self, output_dir = None, state_dict=None):
        return self.save_state(output_dir)

    def train(self, train_dataset, eval_dataset=None, *args, **kwargs):
        train_loader = self.get_train_loader(train_dataset)
        train_loader = self.accelerator.prepare(train_loader)

        if train_dataset.item_feat_dataset is not None:
            item_loader = self.accelerator.prepare(self.get_item_loader(train_dataset.item_feat_dataset))
        else:
            item_loader = None
        
        if eval_dataset is not None:
            eval_loader = self.get_eval_loader(eval_dataset)
            eval_loader = self.accelerator.prepare(eval_loader)
        else:
            eval_loader = None

        for callback in self.callbacks:
            callback_output = callback.on_train_begin(train_dataset, eval_dataset, *args, **kwargs)
            if callback_output.save_checkpoint is not None:
                self.save_state(callback_output.save_checkpoint)
        
        stop_training = False
        try:
            for epoch in range(self.config.epochs):
                epoch_total_loss = 0.0
                epoch_total_bs = 0
                self.model.train()

                self.log_info(f"Start training epoch {epoch}")
                for step, batch in enumerate(train_loader):
    
                    # check if need to do evaluation
                    if (eval_loader is not None) and self._check_if_eval(epoch, self.global_step):
                        metrics = self.evaluation_loop(eval_loader, item_loader)
                        self.log_info("Evaluation at Epoch {} Step {}:".format(epoch, self.global_step))
                        self.log_dict(metrics)

                        for callback in self.callbacks:
                            callback_output = callback.on_eval_end(epoch, self.global_step, metrics, *args, **kwargs)
                            if callback_output.save_checkpoint is not None:
                                self.save_state(callback_output.save_checkpoint)
                            if callback_output.stop_training and (not stop_training):
                                stop_training = True

                    # train one batch
                    batch_size = batch[list(batch.keys())[0]].shape[0]
                    loss_dict = self._train_batch(batch, item_loader=item_loader, *args, **kwargs)
                    loss = loss_dict['loss']

                    self.accelerator.backward(loss)

                    # gradient accumulation and gradient clipping by norm
                    if self.config.gradient_accumulation_steps is None or self.config.gradient_accumulation_steps == 1:
                        self.gradient_clipping(self.config.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    else:
                        if (self.global_step+1) % self.config.gradient_accumulation_steps == 0:
                            self.gradient_clipping(self.config.max_grad_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    epoch_total_loss += loss.item() * batch_size
                    epoch_total_bs += batch_size
                    if (self.global_step % self.config.logging_steps == 0):
                        mean_total_loss = epoch_total_loss / epoch_total_bs
                        self.log_info(f"Epoch {epoch}/{self.config.epochs-1} Step {self.global_step}: Loss {loss:.5f}, Mean Loss {mean_total_loss:.5f}")
                        if (len(loss_dict) > 1):
                            self.log_info(f"\tloss info: ", ', '.join([f'{k}={v:.5f}' for k, v in loss_dict.items()]))

                    for callback in self.callbacks:
                        callback_output = callback.on_batch_end(
                            epoch = epoch,
                            step = self.global_step,
                            logs = loss_dict
                        )
                        if callback_output.save_checkpoint is not None:
                            self.save_state(callback_output.save_checkpoint)
                        if callback_output.stop_training and (not stop_training):
                            stop_training = True
                
                    self.global_step += 1
                    self.cur_global_step += 1

                    if stop_training:
                        self.log_info("[Earlystop] Stop training at epoch {}, {} global steps:".format(epoch, self.global_step))
                        break

                # print loss info at the end of each epoch
                mean_total_loss = epoch_total_loss / epoch_total_bs
                self.log_info(f"Epoch {epoch}/{self.config.epochs} Step {self.global_step}: Loss {loss:.5f}, Mean Loss {mean_total_loss:.5f}")
                if len(loss_dict) > 1:
                    self.log_info(f"\tloss info: ", ', '.join([f'{k}={v:.5f}' for k, v in loss_dict.items()]))
                
                for callback in self.callbacks:
                    callback_output = callback.on_epoch_end(epoch, self.global_step, *args, **kwargs)
                    if callback_output.save_checkpoint is not None:
                        self.save_state(callback_output.save_checkpoint)
                    if callback_output.stop_training and (not stop_training):
                        stop_training = True

                if stop_training:
                    break

                self._total_train_samples = epoch_total_bs
        
        except KeyboardInterrupt:
            self.log_info(f"[KeyboardInterrupt] Stop training at {self.global_step} steps")

        self.log_info(f"[Finished] Stop training at {self.global_step} steps")

        for callback in self.callbacks:
            callback_output = callback.on_train_end(checkpoint_dir=self.config.checkpoint_dir, *args, **kwargs)
            if callback_output.save_checkpoint is not None:
                self.save_state(callback_output.save_checkpoint)

        if eval_loader is not None:
            self.log_info("Start final evaluation...")
            metrics = self.evaluation_loop(eval_loader, item_loader)
            self.log_info("Evaluation result:")
            self.log_dict(metrics)
            for callback in self.callbacks:
                if isinstance(callback, EarlyStopCallback):
                    stop_training = callback.on_eval_end(epoch, self.global_step, metrics, *args, **kwargs)
                else:
                    callback.on_eval_end(epoch, self.global_step, *args, **kwargs)
        
        self.save_state(self.config.checkpoint_dir)

    def load_config(self, config: Union[Dict, str]) -> TrainingArguments:
        if config is None:
            return TrainingArguments()
        if isinstance(config, TrainingArguments):
            return config
        if isinstance(config, dict):
            config_dict = config
        elif isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Config should be either a dictionary or a path to a JSON file, got {type(config)} instead.")
        return TrainingArguments.from_dict(config_dict)

    def get_callbacks(self):
        callbacks = []
        if self.config.earlystop_metric is not None:
            callbacks.append(EarlyStopCallback(
                monitor_metric=self.config.earlystop_metric,
                strategy=self.config.earlystop_strategy,
                patience=self.config.earlystop_patience,
                maximum="max" in self.config.earlystop_metric_mode,
                save=self.config.checkpoint_best_ckpt,
                checkpoint_dir=self.config.checkpoint_dir,
                is_main_process=self.accelerator.is_main_process
            ))
        if self.config.checkpoint_steps is not None:
            callbacks.append(CheckpointCallback(
                step_interval=self.config.checkpoint_steps,
                checkpoint_dir=self.config.checkpoint_dir,
                is_main_process=self.accelerator.is_main_process
            ))
        return callbacks
        
    def get_train_loader(self, train_dataset: Optional[Union[Dataset, str]]=None):
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            train_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.train_dataset[train_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.train_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size)
        return loader
    
    def get_eval_loader(self, eval_dataset: Optional[Union[Dataset, str]]=None):
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            train_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.train_dataset[train_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.train_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        loader = DataLoader(eval_dataset, batch_size=self.config.eval_batch_size)
        return loader
    
    def get_item_loader(self, eval_dataset: Optional[Union[Dataset, str]]=None):
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            train_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.train_dataset[train_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.train_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        loader = DataLoader(eval_dataset, batch_size=self.config.item_batch_size)
        return loader


    @torch.no_grad()
    def evaluation(self, eval_dataset, *args, **kwargs):
        self.log_info("Start evaluation...")
        self.model.eval()
        item_loader = self.get_item_loader(eval_dataset.item_feat_dataset) if eval_dataset.item_feat_dataset is not None else None
        eval_loader = self.get_eval_loader(eval_dataset)
        eval_loader = self.get_eval_loader(eval_dataset)
        metrics = self.evaluation_loop(eval_loader, item_loader)
        self.log_info("Evaluation result:")
        self.log_dict(metrics)


    @torch.no_grad()
    def evaluation_loop(self, eval_loader, item_loader, *args, **kwargs) -> Dict:
        self.model.eval()
        if self.model_type == "retriever":
            self.item_vectors, self.item_ids = self.update_item_vectors(item_loader)
        eval_outputs = []
        eval_total_bs = 0
        for eval_step, eval_batch in enumerate(eval_loader):
            eval_batch_size = eval_batch[list(eval_batch.keys())[0]].shape[0]
            metrics = self._eval_batch(eval_batch, item_vectors=self.item_vectors, *args, **kwargs)
            eval_outputs.append((metrics, eval_batch_size))
            eval_total_bs += eval_batch_size
        metrics = self.eval_epoch_end(eval_outputs)
        self._total_eval_samples = eval_total_bs

        return metrics

    
    @torch.no_grad()
    def update_item_vectors(self, item_loader):
        self.model.eval()
        all_item_vectors, all_item_ids = [], []
        model = self.accelerator.unwrap_model(self.model)
        for item_batch in item_loader:
            item_vector = model.item_encoder(item_batch)
            all_item_vectors.append(item_vector)
            all_item_ids.append(item_batch[model.fiid])
        all_item_vectors = self.accelerator.gather_for_metrics(all_item_vectors)
        all_item_ids = self.accelerator.gather_for_metrics(all_item_ids)
        all_item_vectors = torch.cat(all_item_vectors, dim=0)
        all_item_ids = torch.cat(all_item_ids, dim=0).cpu()
        return all_item_vectors, all_item_ids


    def _check_if_eval(self, epoch, step):
        if self.config.evaluation_strategy == 'epoch':
            if (epoch % self.config.eval_interval == 0) and (self._last_eval_epoch != epoch) and (epoch != 0):
                # do not valid before the first epoch
                self._last_eval_epoch = epoch
                return True
            return False
            # if (epoch % self.config.eval_interval == 0) and (self._last_eval_epoch != epoch):
            #     # do not valid before the first epoch
            #     self._last_eval_epoch = epoch
            #     return True
            # return False
        elif self.config.evaluation_strategy == 'step':
            if step % self.config.eval_interval == 0:
                return True
            return False
        else:
            raise ValueError(f'Unknown evaluation strategy: {self.config.evaluation_strategy}')
    

    def _train_batch(self, batch, *args, **kwargs):
        loss_dict = self.model(batch=batch, cal_loss=True, *args, **kwargs)
        return loss_dict


    @torch.no_grad()
    @staticmethod
    def compute_metrics(metrics, model_type, cutoffs, output: Tuple):
        """ Compute the metrics given the output of the model.

        Args:
            output (Tuple): The output of the model.

        Returns:
            Dict: The computed metrics.
        """
        metrics: list = get_eval_metrics(metrics, model_type)
        output_dict = {}
        if model_type == "retriever":
            for metric, func in metrics:
                for cutoff in cutoffs:
                    output_dict[f"{metric}@{cutoff}"] = func(*output, cutoff)
        else:
            output_dict = (output[0].cpu(), output[1].cpu())    # (pred, target)
        return output_dict


    @torch.no_grad()
    def _eval_batch(self, batch, *args, **kwargs) -> Dict:
        """ Evaluate the model on a batch, return metrics.

        Args:
            batch (Dict): The input batch.

        Returns:
            Dict: The metrics.
        """
        self.model.eval()
        k = max(self.config.cutoffs) if self.config.cutoffs is not None else None
        model = self.accelerator.unwrap_model(self.model)
        outputs = model.eval_step(batch, k=k, *args, **kwargs)
        outputs = self.accelerator.gather_for_metrics(outputs)
        metrics = RetrieverTrainer.compute_metrics(self.config.metrics, self.model_type, self.config.cutoffs, outputs)
        return metrics
    

    @torch.no_grad()
    def eval_epoch_end(self, outputs: List[Tuple]) -> Dict:
        """ Aggregate the metrics from the evaluation batch.

        Args:
            outputs (List): The output of the evaluation batch. It is a list of tuples, 
                where the first element is the metrics (Dict) and the second element is the batch size.

        Returns:
            Dict: The aggregated metrics.
        """
        if self.model_type == "retriever":
            metric_list, bs = zip(*outputs)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                out[k] = ((metric * bs).sum() / bs.sum()).item()
            return out
        else:
            # ranker: AUC, Logloss
            out = {}
            output, bs = zip(*outputs)
            pred, target = zip(*output)
            pred = torch.cat(pred, dim=-1)
            target = torch.cat(target, dim=-1)
            metrics: list = get_eval_metrics(self.config.metrics, self.model_type)
            for metric, func in metrics:
                out[metric] = func(pred, target)
                out[metric] = out[metric].item() if isinstance(out[metric], torch.Tensor) else out[metric]
            return out


    def log_dict(self, d: Dict):
        """Log a dictionary of values."""
        output_list = [f"{k}={v}" for k, v in d.items()]
        self.log_info(", ".join(output_list))

    def log_info(self, *arg, **kwargs):
        if self.accelerator.is_main_process:
            self.logger.info(*arg, **kwargs)


    def get_optimizer(self, name, params, lr, weight_decay):
        r"""Return optimizer for specific parameters.

        .. note::
            If no learner is assigned in the configuration file, then ``Adam`` will be used.

        Args:
            params: the parameters to be optimized.

        Returns:
            torch.optim.optimizer: optimizer according to the config.
        """
        learning_rate = lr
        decay = weight_decay
        if name.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=decay)
        elif name.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def get_lr_scheduler(self):
        return None


    def save_state(self, checkpoint_dir: str):
        """Save the state of the trainer.
        
        Args:
            checkpoint_dir(str): the directory where the checkpoint should be saved.
        """
        # save the parameters of the model
        # save model configuration, enabling loading the model later
        if self.accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save(checkpoint_dir)

            if self.model_type == "retriever":
                item_vectors_path = os.path.join(checkpoint_dir, 'item_vectors.pt')
                torch.save({'item_vectors': self.item_vectors, 'item_ids': self.item_ids}, item_vectors_path)

            # save the optimizer state and the scheduler state
            optimizer_state = {"optimizer": self.optimizer.state_dict()}
            if self.lr_scheduler is not None:
                optimizer_state["scheduler"] = self.scheduler.state_dict()
            torch.save(optimizer_state, os.path.join(checkpoint_dir, 'optimizer_state.pt'))

            # save the trainer configurations
            with open(os.path.join(checkpoint_dir, 'trainer_config.json'), 'w') as fp:
                json.dump(self.config.to_dict(), fp, indent=4)
            # print("self.state:\n",self.state)
            # print("help(self.state):\n",help(self.state))
            # save the trainer state
            # with open(os.path.join(checkpoint_dir, 'trainer_state.json'), 'w') as fp:
            #     json.dump(self.state, fp, indent=4)
            self.state.save_to_json(os.path.join(checkpoint_dir, 'trainer_state.json'))
            self.log_info(f"Saved the model and trainer state to {checkpoint_dir}.")


    # @property
    # def state(self):
    #     state_dict = {
    #         "global_step": self.global_step,
    #     }
    #     return state_dict


    def _check_checkpoint_dir(self):
        if not self.train_mode:
            return
        checkpoint_dir = self.config.checkpoint_dir
        if checkpoint_dir is None:
            raise ValueError("Checkpoint directory must be specified.")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # else:
        #     # check if the checkpoint_dir is empty
        #     if len(os.listdir(checkpoint_dir)) > 0:
        #         raise ValueError(f"Checkpoint directory `{checkpoint_dir}` is not empty.")

    def gradient_clipping(self, clip_norm):
        if (clip_norm is not None) and (clip_norm > 0):
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            self.accelerator.clip_grad_norm_(self.model.parameters(), clip_norm)


    def parameter_init(self, method, **kwargs):
        if method == "xavier_uniform":
            torch.init.xavier_uniform_(self.model.parameters(), **kwargs)
        elif method == "xavier_normal":
            torch.init.xavier_normal_(self.model.parameters(), **kwargs)
        elif method == "orthogonal":
            torch.init.orthogonal_(self.model.parameters(), **kwargs)
        elif method == "constant":
            torch.init.constant_(self.model.parameters(), **kwargs)
        elif method == "zeros":
            torch.init.zeros_(self.model.parameters(), **kwargs)
        else:
            raise ValueError(f"Unknown initialization method: {method}")


__all__ = [
    "metric_dict",
    "get_retriever_metrics",
    "get_ranker_metrics",
    "get_global_metrics",
    "get_eval_metrics"
]


def recall(pred, target, k_or_thres):
    r"""Calculating recall.

    Recall value is defined as below:

    .. math::
        Recall= \frac{TP}{TP+FN}

    Args:
        pred(torch.BoolTensor): [B, num_items] or [B]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target] or [B]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    if pred.dim() > 1:
        k = k_or_thres
        count = (target > 0).sum(-1)
        output = pred[:, :k].sum(dim=-1).float() / count
        return output.mean()
    else:
        thres = k_or_thres
        return M.recall(pred, target, task='binary', threshold=thres)


def precision(pred, target, k_or_thres):
    r"""Calculate the precision.

    Precision are defined as:

    .. math::
        Precision = \frac{TP}{TP+FP}

    Args:
        pred(torch.BoolTensor): [B, num_items] or [B]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target] or [B]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    if pred.dim() > 1:
        k = k_or_thres
        output = pred[:, :k].sum(dim=-1).float() / k
        return output.mean()
    else:
        thres = k_or_thres
        return M.precision(pred, target, task='binary', threshold=thres)


def f1(pred, target, k_or_thres):
    r"""Calculate the F1.

    Args:
        pred(torch.BoolTensor): [B, num_items] or [B]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target] or [B]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    if pred.dim() > 1:
        k = k_or_thres
        count = (target > 0).sum(-1)
        output = 2 * pred[:, :k].sum(dim=-1).float() / (count + k)
        return output.mean()
    else:
        thres = k_or_thres
        return M.f1_score(pred, target, task='binary', threshold=thres)


def map(pred, target, k):
    r"""Calculate the mean Average Precision(mAP).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    count = (target > 0).sum(-1)
    pred = pred[:, :k].float()
    output = pred.cumsum(dim=-1) / torch.arange(1, k+1).type_as(pred)
    output = (output * pred).sum(dim=-1) / \
        torch.minimum(count, k*torch.ones_like(count))
    return output.mean()


def _dcg(pred, k):
    k = min(k, pred.size(1))
    denom = torch.log2(torch.arange(k).type_as(pred) + 2.0).view(1, -1)
    return (pred[:, :k] / denom).sum(dim=-1)


def ndcg(pred, target, k):
    r"""Calculate the Normalized Discounted Cumulative Gain(NDCG).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    pred_dcg = _dcg(pred.float(), k)
    #TODO replace target>0 with target
    ideal_dcg = _dcg(torch.sort((target > 0).float(), descending=True)[0], k)
    all_irrelevant = torch.all(target <= sys.float_info.epsilon, dim=-1)
    pred_dcg[all_irrelevant] = 0
    pred_dcg[~all_irrelevant] /= ideal_dcg[~all_irrelevant]
    return pred_dcg.mean()


def mrr(pred, target, k):
    r"""Calculate the Mean Reciprocal Rank(MRR).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    row, col = torch.nonzero(pred[:, :k], as_tuple=True)
    row_uniq, counts = torch.unique_consecutive(row, return_counts=True)
    idx = torch.zeros_like(counts)
    idx[1:] = counts.cumsum(dim=-1)[:-1]
    first = col.new_zeros(pred.size(0)).scatter_(0, row_uniq, col[idx]+1)
    output = 1.0 / first
    output[first == 0] = 0
    return output.mean()


def hits(pred, target, k):
    r"""Calculate the Hits.

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    return torch.any(pred[:, :k] > 0, dim=-1).float().mean()


def logloss(pred, target):
    r"""Calculate the log loss (log cross entropy).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    if pred.dim() == target.dim():
        return F.binary_cross_entropy_with_logits(pred, target.float())
    else:
        return F.cross_entropy(pred, target)


def auc(pred, target):
    r"""Calculate the global AUC.

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    target = target.type(torch.long)
    return M.auroc(pred, target, task='binary')


def accuracy(pred, target, thres=0.5):
    r"""Calculate the accuracy.

    Args:
        pred(torch.BoolTensor): [Batch_size]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [Batch_size]. The ground truth.

        thres(float): Predictions below the thres will be marked as 0, otherwise 1.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    return M.accuracy(pred, target, task='binary', threshold=thres)


def mse(pred, target):
    """Calculate Meas Square Error"""
    return M.mean_squared_error(pred, target)


def mae(pred, target):
    """Calculate Mean Absolute Error."""
    return M.mean_absolute_error(pred, target)


metric_dict = {
    'ndcg': ndcg,
    'precision': precision,
    'recall': recall,
    'map': map,
    'hit': hits,
    'mrr': mrr,
    'f1': f1,
    'mse': mse,
    'mae': mae,
    'auc': auc,
    'logloss': logloss,
    'accuracy': accuracy
}


def get_retriever_metrics(metric):
    if not isinstance(metric, list):
        metric = [metric]
    topk_metrics = {'ndcg', 'precision', 'recall', 'map', 'mrr', 'hit', 'f1'}
    rank_m = [(m, metric_dict[m])
              for m in metric if m in topk_metrics and m in metric_dict]
    return rank_m


def get_ranker_metrics(metric):
    if not isinstance(metric, list):
        metric = [metric]
    pred_metrics = {'mae', 'mse', 'auc', 'logloss', 'accuracy',
                    'precision', 'recall', 'f1'}
    pred_m = [(m, metric_dict[m]) for m in metric if m in pred_metrics and m in metric_dict]
    return pred_m


def get_global_metrics(metric):
    if (not isinstance(metric, list)) and (not isinstance(metric, dict)):
        metric = [metric]
    global_metrics = {"auc"}
    global_m = [(m, metric_dict[m]) for m in metric if m in global_metrics and m in metric_dict]
    return global_m


def get_eval_metrics(metric_names: Union[List[str], str], model_type: str) -> List[Tuple[str, Callable]]:
    r""" Get metrics with cutoff for evaluation.

    Args:
        metrics_names(Union[List[str], str]): names of metrics which requires cutoff. Such as ["ndcg", "recall"].
        model_type(str): type of model, such as "ranker" or "retriever".

    Returns:
        List[str, Callable[[torch.tensor, torch.tensor], float]]: list of metrics and functions.
    """
    metric_names = metric_names if isinstance(metric_names, list) else [metric_names]
    if model_type == "retriever":
        metrics = get_retriever_metrics(metric_names)
    else:
        metrics = get_ranker_metrics(metric_names)
    return metrics
