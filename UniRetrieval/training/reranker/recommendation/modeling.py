from dataclasses import dataclass
import os
import json
from typing import Dict, Union, Tuple
import faiss.contrib.torch_utils
from typing import OrderedDict
import torch

from UniRetrieval.abc.training.reranker import AbsRerankerModel, RerankerOutput
from UniRetrieval.training.reranker.recommendation.arguments import DataAttr4Model, ModelArguments
from UniRetrieval.modules import MLPModule, LambdaModule, MultiFeatEmbedding, AverageAggregator
from UniRetrieval.modules.arguments import get_model_cls, get_modules, split_batch


class BaseRanker(AbsRerankerModel):
    def __init__(
            self, 
            data_config: DataAttr4Model,
            model_config: Union[Dict, str],
            *args, **kwargs
        ):
        # from BaseModel
        
        self.data_config: DataAttr4Model = data_config
        self.model_config: ModelArguments = self.load_config(model_config)
        self.num_seq_feat = sum([len(seq_feats) for seq_feats in data_config.seq_features.values()])
        self.num_context_feat = len(data_config.context_features)
        self.num_item_feat = len(data_config.item_features)
        self.num_feat = self.num_seq_feat + self.num_context_feat + self.num_item_feat
        super(BaseRanker, self).__init__(*args, **kwargs)
        
        self.model_type = "ranker"
        self.num_items: int = self.data_config.num_items
        self.fiid: str = self.data_config.fiid  # item id field
        self.flabel: str = self.set_labels()
        self.init_modules()
        
    def set_labels(self) -> str:
        # label fields, base ranker only support one label
        # if need multiple labels, use MultiTaskRanker instead
        self.flabel: str = self.data_config.flabels[0]
        return self.data_config.flabels[0]
    
        
    def init_modules(self):
        super().init_modules()
        self.embedding_layer = self.get_embedding_layer()
        self.sequence_encoder = self.get_sequence_encoder()
        self.feature_interaction_layer = self.get_feature_interaction_layer()
        self.prediction_layer = self.get_prediction_layer()


    def get_embedding_layer(self):
        emb = MultiFeatEmbedding(
            features=list(self.data_config.stats.keys()),
            stats=self.data_config.stats,
            embedding_dim=self.model_config.embedding_dim,
            concat_embeddings=False,
            stack_embeddings=True
        )
        return emb
    
    def get_sequence_encoder(self):
        raise NotImplementedError


    def get_feature_interaction_layer(self):
        raise NotImplementedError

    
    def get_prediction_layer(self):
        raise NotImplementedError

    def get_loss_function(self):
        # BCELoss is not good for autocast in distributed training, reminded by pytorch
        return get_modules("loss", "BCEWithLogitLoss")(reduction='mean')


    def compute_score(
            self, 
            batch, 
            *args, 
            **kwargs
        ) -> RerankerOutput:
        context_feat, item_feat, seq_feat_dict = split_batch(batch, self.data_config)
        all_embs = []
        context_emb = self.embedding_layer(context_feat, strict=False)  # [B, N2, D]
        item_emb = self.embedding_layer(item_feat, strict=False)    # [B, N3, D]
        if len(seq_feat_dict) > 0:
            for seq_name, seq_feat in seq_feat_dict.items():
                padding_mask = seq_feat[self.fiid] == 0
                seq_emb = self.embedding_layer(seq_feat, strict=False)
                seq_rep = self.sequence_encoder[seq_name](
                    seq=seq_emb,
                    padding_mask=padding_mask,
                    target=item_emb,
                )   # [B, N1, D]
                all_embs.append(seq_rep)
        all_embs += [context_emb, item_emb]
        all_embs = torch.concat(all_embs, dim=1) # [B, N1+N2+N3, D]
        interacted_emb = self.feature_interaction_layer(all_embs)    # [B, **]
        score = self.prediction_layer(interacted_emb)   # [B], sigmoid
        if len(score.shape) == 2 and score.size(-1) == 1:
            score = score.squeeze(-1)   # [B, 1] -> [B]
        return RerankerOutput(score, [context_emb, item_emb, seq_emb])
    
    
    def get_score_function(self):
        return self.compute_score
    
    def sampling(self, query, num_neg, *args, **kwargs):
        return self.negative_sampler(query, num_neg)
        
    def forward(self, batch, cal_loss=False, *args, **kwargs) -> RerankerOutput:
        if cal_loss:
            return self.compute_loss(batch, *args, **kwargs)
        else:
            output = self.compute_score(batch, *args, **kwargs)
        return output

    def compute_loss(self, batch, *args, **kwargs) -> Dict:
        label = batch[self.flabel].float()
        output = self.forward(batch, *args, **kwargs)
        output_dict = output.to_dict()
        output_dict['label'] = label
        loss = self.loss_function(**output_dict)
        
        if isinstance(loss, dict):
            return loss
        else:
            return {'loss': loss}
        
    @torch.no_grad()
    def eval_step(self, batch, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.compute_score(batch, *args, **kwargs)
        score = output.score
        # check if the last layer of the model is a sigmoid function
        # get the last layer of the model
        pred = score
        target = batch[self.flabel].float()
        return pred, target

    @torch.no_grad()
    def predict(self, context_input: Dict, candidates: Dict, topk: int=6, gpu_mem_save=False, *args, **kwargs):
        """ predict topk candidates for each context
        
        Args:
            context_input (Dict): input context feature
            candidates (Dict): candidate items
            topk (int): topk candidates
            gpu_mem_save (bool): whether to save gpu memroy by using loop to process each candidate

        Returns:
            torch.Tensor: topk indices (offset instead of real item id)
        """
        num_candidates = candidates[self.fiid].size(1)
        if not gpu_mem_save:
            # expand batch to match the number of candidates, consuming more memory
            batch_size = candidates[self.fiid].size(0)
            for k, v in context_input.items():
                # B, * -> BxN, *
                if isinstance(v, dict):
                    for k_, v_ in v.items():
                        # 替代 repeat_interleave
                        v_expanded = v_.unsqueeze(1).expand(-1, num_candidates, *v_.shape[1:])  # 扩展
                        v[k_] = v_expanded.reshape(-1, *v_.shape[1:])  # 展平
                else:
                    # 替代 repeat_interleave
                    v_expanded = v.unsqueeze(1).expand(-1, num_candidates, *v.shape[1:])  # 扩展
                    context_input[k] = v_expanded.reshape(-1, *v.shape[1:])  # 展平
                    
            for k, v in candidates.items():
                # B, N, * -> BxN, *
                candidates[k] = v.view(-1, *v.shape[2:])
            context_input.update(candidates)    # {key: BxN, *}
            output = self.compute_score(context_input, *args, **kwargs)
            scores = output.score.view(batch_size, num_candidates)  # [B, N]
        else:
            # use loop to process each candidate
            scores = []
            for i in range(num_candidates):
                candidate = {k: v[:, i] for k, v in candidates.items()}
                new_batch = dict(**context_input)
                new_batch.update(candidate)
                output = self.compute_score(new_batch, *args, **kwargs)
                scores.append(output.score)
            scores = torch.stack(scores, dim=-1)    # [B, N]
        
        # get topk idx
        topk = min(self.model_config.topk, num_candidates)
        topk_score, topk_idx = torch.topk(scores, topk)
        return topk_idx

    
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
        data_config = DataAttr4Model.from_dict(config_dict['data_config'])
        model_cls = get_model_cls(config_dict['model_type'], config_dict['model_name'])
        del config_dict['data_config'], config_dict['model_type'], config_dict['model_name']
        
        model_config = ModelArguments.from_dict(config_dict)
        ckpt_path = os.path.join(checkpoint_dir, "model.pt")
        state_dict = torch.load(ckpt_path, weights_only=True)
        model = model_cls(data_config, model_config)
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
        config_dict['model_name_or_path'] = checkpoint_dir
        config_dict['model_type'] = self.model_type
        config_dict['model_name'] = self.__class__.__name__
        config_dict['data_config'] = self.data_config.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)


class MLPRanker(BaseRanker):
    def __init__(self, ranker_data_config, ranker_model_config, *args, **kwargs):
        super().__init__(ranker_data_config, ranker_model_config, *args, **kwargs)
        
    def get_sequence_encoder(self):
        encoder_dict = torch.nn.ModuleDict({
            name: AverageAggregator(dim=1)
            for name in self.data_config.seq_features
        })
        return encoder_dict
        
    def get_feature_interaction_layer(self):
        flatten_layer = LambdaModule(lambda x: x.flatten(start_dim=1))  # [B, N, D] -> [B, N*D]
        mlp_layer = MLPModule(
            mlp_layers= [self.num_feat * self.model_config.embedding_dim] + self.model_config.mlp_layers,
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        return torch.nn.Sequential(flatten_layer, mlp_layer)
    

    def get_prediction_layer(self):
        pred_mlp = MLPModule(
            mlp_layers=[self.model_config.mlp_layers[-1]] + self.model_config.prediction_layers + [1],
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        return pred_mlp
    
class MyMLPRanker(MLPRanker):
    def __init__(self, ranker_data_config, ranker_model_config, *args, **kwargs):
        super().__init__(ranker_data_config, ranker_model_config, *args, **kwargs)
        self.topk=6 

    def compute_score(self, batch, *args, **kwargs) -> RerankerOutput:
        context_feat, seq_feat, item_feat = split_batch(batch, self.data_config)
        all_embs = []
        if len(seq_feat) > 0:
            seq_emb = self.embedding_layer(seq_feat, strict=False)
            seq_rep = self.sequence_encoder(seq_emb)   # [B, N1, D]
            all_embs.append(seq_rep)
        context_emb = self.embedding_layer(context_feat, strict=False)  # [B, N2, D]
        item_emb = self.embedding_layer(item_feat, strict=False)    # [B, N3, D]
        all_embs += [context_emb, item_emb]
        all_embs = torch.concat(all_embs, dim=1) # [B, N1+N2+N3, D]
        interacted_emb = self.feature_interaction_layer(all_embs)    # [B, **]
        score = self.prediction_layer(interacted_emb)   # [B], sigmoid
        if len(score.shape) == 2 and score.size(-1) == 1:
            score = score.squeeze(-1)   # [B, 1] -> [B]
        return RerankerOutput(score, [context_emb, item_emb, seq_emb])
    
    def forward(self, batch, cal_loss=False, *args, **kwargs) -> RerankerOutput:
        output = self.compute_score(batch, *args, **kwargs)
        return output
    
    @torch.no_grad()
    def predict(self, context_input: Dict, candidates: Dict, topk: int=6, gpu_mem_save=False, *args, **kwargs):
        """ predict topk candidates for each context
        
        Args:
            context_input (Dict): input context feature
            candidates (Dict): candidate items
            topk (int): topk candidates
            gpu_mem_save (bool): whether to save gpu memroy by using loop to process each candidate

        Returns:
            torch.Tensor: topk indices (offset instead of real item id)
        """
        num_candidates = candidates[self.fiid].size(1)
        
        if not gpu_mem_save:
            self.topk=1
            batch_size = candidates[self.fiid].size(0)
            for k, v in context_input.items():
                # B, * -> BxN, *
                if isinstance(v, dict):
                    for k_, v_ in v.items():
                        # 替代 repeat_interleave
                        v_expanded = v_.unsqueeze(1).expand(-1, num_candidates, *v_.shape[1:])  # 扩展
                        v[k_] = v_expanded.reshape(-1, *v_.shape[1:])  # 展平
                else:
                    # 替代 repeat_interleave
                    v_expanded = v.unsqueeze(1).expand(-1, num_candidates, *v.shape[1:])  # 扩展
                    context_input[k] = v_expanded.reshape(-1, *v.shape[1:])  # 展平
                    
            for k, v in candidates.items():
                # B, N, * -> BxN, *
                candidates[k] = v.view(-1, *v.shape[2:])
            context_input.update(candidates)    # {key: BxN, *}
            output = self.compute_score(context_input, *args, **kwargs)
            scores = output.score.view(batch_size, num_candidates)
            
        else:
            scores = []
            for i in range(num_candidates):
                candidate = {k: v[:, i] for k, v in candidates.items()}
                new_batch = dict(**context_input)
                new_batch.update(candidate)
                output = self.compute_score(new_batch, *args, **kwargs)
                scores.append(output.score)
            scores = torch.stack(scores, dim=-1)  
        # get topk idx
        
        self.topk = min(self.topk, num_candidates)
        topk_score, topk_idx = torch.topk(scores, self.topk)
        return topk_idx
    
