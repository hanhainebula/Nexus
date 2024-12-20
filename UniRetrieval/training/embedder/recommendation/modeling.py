from dataclasses import dataclass
import os
import json
from typing import Dict, Union, Tuple
from tqdm import tqdm
import faiss
import faiss.contrib.torch_utils
from typing import OrderedDict
import torch

from UniRetrieval.abc.training.embedder import AbsEmbedderModel, EmbedderOutput
from .arguments import DataAttr4Model, ModelArguments
from .datasets import ItemDataset
from UniRetrieval.modules import QueryEncoder, ItemEncoder, UniformSampler
from UniRetrieval.modules.score import InnerProductScorer
from UniRetrieval.modules.loss import BPRLoss
from UniRetrieval.utils import get_model_cls

@dataclass
class RetrieverModelOutput(EmbedderOutput):
    pos_score: torch.tensor = None
    neg_score: torch.tensor = None
    log_pos_prob: torch.tensor = None
    log_neg_prob: torch.tensor = None
    query_vector: torch.tensor = None
    pos_item_vector: torch.tensor = None
    neg_item_vector: torch.tensor = None
    pos_item_id: torch.tensor = None
    neg_item_id: torch.tensor = None


# TODO 实现data_config合并到model_config中
class BaseRetriever(AbsEmbedderModel):
    def __init__(
            self, 
            data_config: DataAttr4Model,
            model_config: Union[Dict, str],
            *args, **kwargs
        ):
        # from BaseModel
        super(BaseRetriever, self).__init__(*args, **kwargs)
        self.data_config: DataAttr4Model = data_config
        self.model_config: ModelArguments = self.load_config(model_config)
        self.model_type = 'retriever'
        # self.loggers = get_logger(training_config)
        self.init_modules()

        self.num_items: int = self.data_config.num_items
        self.fiid: str = self.data_config.fiid  # item id field
        self.flabel: str = self.data_config.flabels[0]  # label field, only support one label for retriever
        self.item_vectors = None
        self.item_ids = None

    def init_modules(self):
        # super().init_modules()
        self.item_encoder = self.get_item_encoder()
        self.query_encoder = self.get_query_encoder()
        self.score_function = self.get_score_function()
        self.negative_sampler = self.get_negative_sampler()
        self.loss_function = self.get_loss_function()

    def get_query_encoder(self):
        raise NotImplementedError

    def get_item_encoder(self):
        raise NotImplementedError

    def get_score_function(self):
        raise NotImplementedError

    def get_negative_sampler(self):
        return None

    def get_loss_function(self):
        raise NotImplementedError

    def compute_score(
            self, 
            batch,
            inference=False,
            item_loader=None,
            *args, 
            **kwargs
        ):
        pos_item_id = batch[self.fiid]
        query_vec = self.query_encoder(batch)
        pos_item_vec = self.item_encoder(batch)
        pos_score = self.score_function(query_vec, pos_item_vec)

        if not inference:
            # training mode, sampling negative items or use all items as negative items
            if self.negative_sampler:
                # sampling negative items
                if not self.model_config.num_neg:
                    raise ValueError("`negative_count` is required when `sampler` is not none.")
                else:
                    neg_item_idx, log_neg_prob = self.sampling(query_vec, self.model_config.num_neg)
                    neg_item_feat = self.get_item_feat(item_loader.dataset, neg_item_idx)
                    neg_item_id = neg_item_feat.get(self.fiid)
                    neg_item_vec = self.item_encoder(neg_item_feat)
                    # log_pos_prob = self.negative_sampler.compute_item_p(query_vec, pos_item_id)
                    log_pos_prob = None
            else:
                raise NotImplementedError("Full softmax is not supported for industrial dataset yet.")
            neg_score = self.score_function(query_vec, neg_item_vec)
        else:
            neg_score = None
            neg_item_id = None
            neg_item_vec = None
            log_pos_prob, log_neg_prob = None, None

        output = RetrieverModelOutput(
            pos_score=pos_score,
            neg_score=neg_score,
            log_pos_prob=log_pos_prob,
            log_neg_prob=log_neg_prob,
            query_vector=query_vec,
            pos_item_vector=pos_item_vec,
            neg_item_vector=neg_item_vec,
            pos_item_id=pos_item_id,
            neg_item_id=neg_item_id
        )
        return output
    
    def sampling(self, query, num_neg, *args, **kwargs):
        return self.negative_sampler(query, num_neg)
        
    def forward(self, batch, cal_loss=False, *args, **kwargs) -> RetrieverModelOutput:
        if cal_loss:
            return self.compute_loss(batch, *args, **kwargs)
        else:
            output = self.compute_score(batch, *args, **kwargs)
        return output

    def compute_loss(self, batch, *args, **kwargs) -> Dict:
        output = self.forward(batch, *args, **kwargs)
        output_dict = output.to_dict()
        labels = batch[self.flabel]
        output_dict['label'] = labels
        loss = self.loss_function(**output_dict)
        if isinstance(loss, dict):
            return loss
        else:
            return {'loss': loss}
        
    @torch.no_grad()
    def eval_step(self, batch, k, item_vectors, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            query_vec = self.query_encoder(batch)
            pos_vec = self.item_encoder(batch)
            pos_scores = self.score_function(query_vec, pos_vec)
            # TODO: use faiss to reduce memory usage
            res = faiss.StandardGpuResources()
            if self.score_function.__class__.__name__ == 'InnerProductScorer':
                index_flat = faiss.IndexFlatIP(item_vectors.shape[-1])
            elif self.score_function.__class__.__name__ == 'EuclideanScorer':
                index_flat = faiss.IndexFlatL2(item_vectors.shape[-1])
            elif self.score_function.__class__.__name__ == 'CosineScorer':
                item_vectors = torch.nn.functional.normalize(item_vectors, p=2, dim=-1)
                pos_vec = torch.nn.functional.normalize(pos_vec, p=2, dim=-1)
                index_flat = faiss.IndexFlatIP(item_vectors.shape[-1])
            else:
                raise NotImplementedError(f"Not supported scorer {self.score_function.__class__.__name__}.")
            gpu_index = next(self.parameters()).device.index
            index_flat = faiss.index_cpu_to_gpu(res, gpu_index, index_flat)
            index_flat.add(item_vectors)

            topk_scores, topk_indices = index_flat.search(query_vec, k)

            # we usually do not mask history in industrial settings
            if pos_scores.dim() < topk_scores.dim():
                pos_scores = pos_scores.unsqueeze(-1)
            all_scores = torch.cat([pos_scores, topk_scores], dim=-1) # [B, N + 1]
            # sort and get the index of the first item
            _, indice = torch.sort(all_scores, dim=-1, descending=True, stable=True)
            pred = indice[:, :k] == 0
            target = torch.ones_like(batch[self.fiid], dtype=torch.bool).view(-1, 1)   # [B, 1]
            return pred, target

    @torch.no_grad()
    def encode_query(self, context_input: dict) -> torch.Tensor:
        """ Encode context input, output vectors.
        Args:
            context_input (dict): context input features, e.g., user_id, session_id, etc.
        Returns:
            torch.Tensor: [B, D], where B is batch size and D is embedding dimension.
        """
        context_vec = self.query_encoder(context_input)
        return context_vec
    
    @torch.no_grad()
    def predict(self, context_input: Dict, candidates: Dict, topk: int, *args, **kwargs):
        """ predict topk candidates for each context
        
        Args:
            context_input (Dict): input context feature
            candidates (Dict): candidate items
            topk (int): topk candidates

        Returns:
            torch.Tensor: topk indices (offset instead of real item id)
        """
        query_vec = self.query_encoder(context_input)
        candidate_item_vec = self.item_encoder(candidates)  # [B, N, D]

        scores = self.score_function(query_vec, candidate_item_vec)
        # get topk idx
        topk_score, topk_idx = torch.topk(scores, topk)
        return topk_idx

    def get_item_feat(self, item_dataset: ItemDataset, item_id: torch.tensor):
        """
        Get item features by item id.
        Args:
            item_id: [B]
        Returns:
            item_feat: [B, N]
        """
        item_feat = item_dataset.get_item_feat(item_id)
        return item_feat
    
    def load_config(self, config: Union[Dict, str]) -> ModelArguments:
        if isinstance(config, ModelArguments):
            config = config
        elif isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config = ModelArguments.from_dict(config_dict)
        elif isinstance(config, dict):
            config = ModelArguments.from_dict(config)
        else:
            raise ValueError(f"Config should be either a dictionary or a path to a JSON file, got {type(config)} instead.")
        return config


    @staticmethod
    def from_pretrained(checkpoint_dir: str):
        config_path = os.path.join(checkpoint_dir, "model_config.json")
        with open(config_path, "r", encoding="utf-8") as config_path:
            config_dict = json.load(config_path)
        data_attr = DataAttr4Model.from_dict(config_dict['data_attr'])
        model_cls = get_model_cls(config_dict['model_type'], config_dict['model_name'])
        del config_dict['data_attr'], config_dict['model_type'], config_dict['model_name']
        model_config = ModelArguments.from_dict(config_dict)
        ckpt_path = os.path.join(checkpoint_dir, "model.pt")
        state_dict = torch.load(ckpt_path, weights_only=True)
        model = model_cls(data_attr, model_config)
        if "item_vectors" in state_dict:
            model.item_vectors = state_dict["item_vectors"]
            del state_dict['item_vectors']
        model.load_state_dict(state_dict=state_dict, strict=True)
        return model


    def save(self, checkpoint_dir: str, **kwargs):
        self.save_checkpoint(checkpoint_dir)
        self.save_configurations(checkpoint_dir)

    def save_checkpoint(self, checkpoint_dir: str):
        path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.state_dict(), path)
        
    def save_configurations(self, checkpoint_dir: str):
        path = os.path.join(checkpoint_dir, "model_config.json")
        config_dict = self.model_config.to_dict()
        config_dict['model_type'] = self.model_type
        config_dict['model_name'] = self.__class__.__name__
        config_dict['data_attr'] = self.data_config.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

    #TODO to be implemented
    def encode_info(self, *args, **kwargs):
        return super().encode_info(*args, **kwargs)


class MLPRetriever(BaseRetriever):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def get_item_encoder(self):
        return ItemEncoder(self.data_config, self.model_config)
    

    def get_query_encoder(self):
        return QueryEncoder(self.data_config, self.model_config,self.item_encoder)
    

    def get_score_function(self):
        return InnerProductScorer()
    
    def get_loss_function(self):
        return BPRLoss()
    
    def get_negative_sampler(self):
        return UniformSampler(num_items=self.data_config.num_items)
    
    # TODO to be implemented
    def encode_info(self, *args, **kwargs):
        return super().encode_info(*args, **kwargs)