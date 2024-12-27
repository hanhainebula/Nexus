import os
import json
import torch
import itertools

from torch import nn
from loguru import logger as loguru_logger
import torch.optim as optim
import torch.distributed as dist
import torchrec_dynamic_embedding as tde

from typing import cast, Dict, Union, List, Any
from fbgemm_gpu.split_embedding_configs import EmbOptimType

from torchrec import EmbeddingConfig, EmbeddingCollection, KeyedJaggedTensor
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.embedding import EmbeddingCollectionSharder, ShardedEmbeddingCollection
from torchrec.distributed.model_parallel import DistributedModelParallel, get_default_sharders
from torchrec.distributed.planner import Topology, EmbeddingShardingPlanner
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.distributed.types import ModuleSharder, ShardingType, ShardingPlan, ShardingEnv, EmbeddingModuleShardingPlan

from rs4industry.config.training import TrainingArguments

from torchrec_test.data.tde_dataset import get_TDE_dataloader, get_dataloader
from torchrec_test.data.collection_dataloader import collection_wrap


class Model(nn.Module):
    def __init__(self, e_configs):
        super().__init__()
        self.emb = EmbeddingCollection(tables=e_configs, device=torch.device("meta"))
        self.linear_1 = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.MSELoss()
        
    def forward(self, kjt: Batch):
        # call recstudio model
        # recstudio_model_output = recstudio_model(kjt)
        # return recstudio_model(kjt, cal_loss=True), recstudio_model_output
        emb = self.emb(kjt.sparse_features)
        
        pred = torch.sum(torch.multiply(emb[two_tower_column_names[0]].values(), 
                                        emb[two_tower_column_names[1]].values()),
                         dim=-1)
        loss = self.loss_fn(self.sigmoid(pred), kjt.labels.float())
        return loss, {"loss": loss.detach()}


class DemoTrainer(object):
    def __init__(self, model, config=None, train=True):
        super(DemoTrainer, self).__init__()
        self.config: TrainingArguments = TrainingArguments()

        # test config
        self.config.learning_rate = 0.01
        self.config.weight_decay = 0.001
        self.config.optimizer = "adam"
        self.config.train_batch_size = 100
        self.config.epochs = 10

        self.config.ps_url = "redis://127.0.0.1:6379/?prefix=test_junwei"
        # TODO: config sharding types
        self.config.sharding_types = [ShardingType.ROW_WISE.value]

        self.logger = loguru_logger
        self.global_step = 0 # global step counter, load from checkpoint if exists
        self.cur_global_step = 0 # current global step counter, record the steps of this training
        self._last_eval_epoch = -1

        self._total_train_samples = 0
        self._total_eval_samples = 0

        self.rank, self.world_size, self.device, self.backend, self.pg = self._init_process()
        self.embedding_configs_dict = self._get_embedding_configs_dict(model)
        self.path = self._get_embedding_path(model)
        self.embedding_configs = self._get_embedding_configs(model)
        self.feature_names = self._get_feature_names(model)

        topology = Topology(
            world_size=self.world_size,
            local_world_size=self.world_size,
            compute_device=self.device.type,
        )

        fused_params = {
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "optimizer": getattr(EmbOptimType, self.config.optimizer.upper()),
        }

        sharders = self._get_sharders(fused_params=fused_params)

        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=self.config.train_batch_size,
            constraints={
                f"{feature_name}": ParameterConstraints(sharding_types=self.config.sharding_types)
                for feature_name in self.feature_names
            }
        )

        plan: ShardingPlan = planner.collective_plan(model, sharders, self.pg)
        
        self.model = DistributedModelParallel(
            module=model,
            env=ShardingEnv.from_process_group(self.pg),
            device=self.device,
            plan=plan,
            sharders=sharders,
        )

        sharded_modules = self._get_sharded_modules_recursive(self.model.module, "", plan)
        self.sharded_module, self.params_plan = sharded_modules[self.path]

        self.optimizer = CombinedOptimizer([
            self.model.fused_optimizer,
            KeyedOptimizerWrapper(
                dict(in_backward_optimizer_filter(self.model.named_parameters())),
                lambda params: self._get_optimizer(
                    name=self.config.optimizer,
                    params=params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                ),
            ),
        ])

        self.lr_scheduler = None

    def fit(self):
        train_loader = get_dataloader(
            batch_size=self.config.train_batch_size,
            num_embeddings=100000000,
            num_batches=100,
        )

        eval_loader = None

        # train_dataloader, model = tde.wrap(
        #     url=self.config.ps_url,
        #     dataloader=train_dataloader,
        #     module=self.model,
        #     configs_dict=self.embedding_configs_dict,
        # )

        train_loader = collection_wrap(
            url=self.config.ps_url,
            dataloader=train_loader,
            path=self.path,
            configs=self.embedding_configs,
            sharded_module=self.sharded_module,
            params_plan=self.params_plan,
        )

        train_pipeline = TrainPipelineSparseDist(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
        )

        stop_training = False
        try:
            for epoch in range(self.config.epochs):
                epoch_train_loader = iter(train_loader)

                epoch_total_loss = 0.0
                epoch_total_bs = 0
                train_pipeline._model.train()

                self._log_info(f"Start training epoch {epoch}")
                while True:

                    # check if need to do evaluation
                    
                    # train one batch
                    try:
                        loss_dict = train_pipeline.progress(epoch_train_loader)
                        loss = loss_dict["loss"]

                        # gradient accumulation and gradient clipping by norm


                        epoch_total_loss += loss.item()
                        epoch_total_bs += 1
                        if (self.global_step % self.config.logging_steps == 0):
                            mean_total_loss = epoch_total_loss / epoch_total_bs
                            self._log_info(f"Epoch {epoch}/{self.config.epochs - 1} Step {self.global_step}: Loss {loss:.5f}, Mean Loss {mean_total_loss:.5f}")
                            if (len(loss_dict) > 1):
                                self._log_info(f"\tloss info: ", ', '.join([f'{k}={v:.5f}' for k, v in loss_dict.items()]))
                        
                        self.global_step += 1
                        self.cur_global_step += 1

                        if stop_training:
                            self._log_info("[Earlystop] Stop training at epoch {}, {} global steps:".format(epoch, self.global_step))
                            break

                    except StopIteration:
                        break
                
                # print loss info at the end of each epoch
                mean_total_loss = epoch_total_loss / epoch_total_bs
                self._log_info(f"Epoch {epoch}/{self.config.epochs} Step {self.global_step}: Loss {loss:.5f}, Mean Loss {mean_total_loss:.5f}")
                if len(loss_dict) > 1:
                    self._log_info(f"\tloss info: ", ', '.join([f'{k}={v:.5f}' for k, v in loss_dict.items()]))

                if stop_training:
                    break

                self._total_train_samples = epoch_total_bs

        except KeyboardInterrupt:
            self._log_info(f"[KeyboardInterrupt] Stop training at {self.global_step} steps")
        
        self._log_info(f"[Finished] Stop training at {self.global_step} steps")

    def _check_if_eval(self, epoch, step):
        if self.config.evaluation_strategy == 'epoch':
            if (epoch % self.config.eval_interval == 0) and (self._last_eval_epoch != epoch) and (epoch != 0):
                # do not valid before the first epoch
                self._last_eval_epoch = epoch
                return True
            return False
        elif self.config.evaluation_strategy == 'step':
            if step % self.config.eval_interval == 0:
                return True
            return False
        else:
            raise ValueError(f'Unknown evaluation strategy: {self.config.evaluation_strategy}')

    def _log_info(self, *arg, **kwargs):
        if self.rank == 0:
            self.logger.info(*arg, **kwargs)
    
    # initialize distributed training
    def _init_process(self):
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            device: torch.device = torch.device(f"cuda:{rank}")
            backend = "nccl"
            torch.cuda.set_device(device)
        else:
            device: torch.device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.group.WORLD

        return rank, world_size, device, backend, pg
    
    def _get_optimizer(self, name, params, lr, weight_decay):
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

    def _get_sharders(self, fused_params: Dict[str, Any] | None = None) -> List[ModuleSharder[nn.Module]]:
        return [
            cast(ModuleSharder[nn.Module], EmbeddingCollectionSharder(fused_params=fused_params)),
        ]
    
    def _get_embedding_configs_dict(self, model) -> Dict[str, List[EmbeddingConfig]]:
        """Get the embedding configurations from the model.

        Returns:
            Dict[str, List[EmbeddingConfig]]: a dictionary of embedding configurations.
        Example:
            {"emb": [EmbeddingConfig, EmbeddingConfig]}
        """
        embedding_configs_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, EmbeddingCollection):
                embedding_name = name.split(".")[-1]
                embedding_configs_dict[embedding_name] = module._embedding_configs
        return embedding_configs_dict
    
    def _get_embedding_path(self, model) -> str:
        """Get the path of the embedding module in the model.

        Returns:
            str: the path of the embedding module.
        """
        for name, module in model.named_modules():
            if isinstance(module, EmbeddingCollection):
                return name.split(".")[-1]
        return ""
    
    def _get_embedding_configs(self, model) -> List[EmbeddingConfig]:
        """Get the embedding configurations from the model.

        Returns:
            List[EmbeddingConfig]: a list of embedding configurations.
        """
        for _, module in model.named_modules():
            if isinstance(module, EmbeddingCollection):
                return module._embedding_configs
        return []
    
    def _get_sharded_modules_recursive(self, module: nn.Module, path: str, plan: ShardingPlan) -> Dict[str, nn.Module]:
        """
        Get all sharded modules of module from `plan`.
        """
        params_plan = plan.get_plan_for_module(path)
        if params_plan:
            return {path: (module, params_plan)}

        res = {}
        for name, child in module.named_children():
            new_path = f"{path}.{name}" if path else name
            res.update(self._get_sharded_modules_recursive(child, new_path, plan))
        return res
    
    def _get_feature_names(self, model) -> List[str]:
        """Get the feature names from the model.
        
        Returns:
            List[str]: a list of feature names.

        Example:
            ["userId", "movieId", "rating", "timestamp"]
        """
        feature_names = []
        for _, module in model.named_modules():
            if isinstance(module, EmbeddingCollection):
                flatten_feature_names = list(itertools.chain(*[config.feature_names for config in module._embedding_configs]))
                feature_names.extend(flatten_feature_names)

        return feature_names


if __name__ == '__main__':
    from torchrec.datasets.movielens import DEFAULT_RATINGS_COLUMN_NAMES

    two_tower_column_names = DEFAULT_RATINGS_COLUMN_NAMES[:2]
    e_configs = [
        EmbeddingConfig(
            name=f"{feature_name}",
            embedding_dim=512,
            num_embeddings=500,
            feature_names=[feature_name],
            weight_init_min=-0.02,
            weight_init_max=0.02,
        )
        for feature_name in two_tower_column_names
    ]

    model = Model(e_configs)
    trainer = DemoTrainer(model)

    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()
