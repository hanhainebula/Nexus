import os 
from typing import Dict, List, Optional, Tuple, Union, Any, cast
import itertools

import torch 
import torch.distributed as dist

from Nexus.abc.training.reranker import AbsRerankerRunner
from Nexus.modules.optimizer import get_lr_scheduler, get_optimizer
from .arguments import TrainingArguments, ModelArguments, DataArguments, DataAttr4Model
from .modeling import BaseRanker
from .tde_modeling import TDEModel
from .trainer import TDERankerTrainer
from .dataset import AbsRecommenderRerankerCollator, ConfigProcessor, ShardedDataset
from .runner import RankerRunner

from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec import EmbeddingConfig, EmbeddingBagConfig, EmbeddingCollection, EmbeddingBagCollection
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fused_embeddingbag import FusedEmbeddingBagCollectionSharder
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.planner import Topology, EmbeddingShardingPlanner
from torchrec.distributed.model_parallel import DistributedModelParallel, get_default_sharders
from torchrec.distributed.types import ModuleSharder, ShardingType, ShardingPlan, ShardingEnv
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

from dynamic_embedding.wrappers import attach_id_transformer_group

class TDERankerRunner(RankerRunner):
    """
    Finetune Runner for base embedding models.
    """
    def __init__(
        self,
        model_config_or_path: Union[str, ModelArguments],
        data_config_or_path: Union[str, DataArguments],
        train_config_or_path: Union[str, TrainingArguments],
        model_class: BaseRanker,
        model=None,
        trainer=None,
        *args,
        **kwargs,
    ):        
        self.model_class = model_class
        
        self.data_args = DataArguments.from_json(data_config_or_path) if isinstance(data_config_or_path, str) else data_config_or_path
        self.model_args = ModelArguments.from_json(model_config_or_path) if isinstance(model_config_or_path, str) else model_config_or_path
        self.training_args = TrainingArguments.from_json(train_config_or_path) if isinstance(train_config_or_path, str) else train_config_or_path
        
        self.train_dataset, self.cp_attr = self.load_dataset()
        if model is not None: 
            self.model = model 
        else:
            self.model, tde_configs_dict, tde_feature_names, tde_settings = self.load_model()
        self.data_collator = self.load_data_collator()
        self.trainer = trainer if trainer is not None else self.load_trainer(tde_configs_dict, tde_feature_names, tde_settings)
    
    def _prepare_model(self, model):
        
        def _init_process():
            rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            if torch.cuda.is_available():
                device: torch.device = torch.device(f"cuda:{rank}")
                backend = "nccl"
                torch.cuda.set_device(device)
            else:
                device: torch.device = torch.device("cpu")
                backend = "gloo"
            if not dist.is_initialized():
                dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = dist.group.WORLD

            return rank, world_size, device, backend, pg
        
        def _get_tde_configs_dict(model) -> Dict:
            """Get the embedding configurations from the model.

            Returns:
                Dict[str, List[EmbeddingConfig]]: a dictionary of embedding configurations.
            Example:
                {"emb": [EmbeddingConfig, EmbeddingConfig], "emb2": [EmbeddingConfig, EmbeddingConfig]}
            """
            embedding_configs_dict = {}
            for name, module in model.named_modules():
                if isinstance(module, (EmbeddingCollection, EmbeddingBagCollection)):
                    embedding_configs_dict[name] = module._embedding_configs
            return embedding_configs_dict
    
        def _get_tde_feature_names(model) -> List[str]:
            """Get the feature names from the model.
            
            Returns:
                List[str]: a list of feature names.

            Example:
                ["userId", "movieId", "rating", "timestamp"]
            """
            feature_names = []
            for _, module in model.named_modules():
                if isinstance(module, (EmbeddingCollection, EmbeddingBagCollection)):
                    flatten_feature_names = list(itertools.chain(*[config.feature_names for config in module._embedding_configs]))
                    feature_names.extend(flatten_feature_names)
            return feature_names
        
        def _get_sharders(fused_params: Dict[str, Any] | None = None):
            return [
                EmbeddingBagCollectionSharder(fused_params=fused_params),
                FusedEmbeddingBagCollectionSharder(fused_params=fused_params),
                EmbeddingCollectionSharder(fused_params=fused_params),
            ]
        
        model = TDEModel(model)
        # get tde configs 
        tde_configs_dict = _get_tde_configs_dict(model)
        tde_feature_names = _get_tde_feature_names(model)
        tde_settings = model.model_config.tde_settings
        
        rank, world_size, device, backend, pg = _init_process()
        topology = Topology(
            world_size=world_size,
            local_world_size=world_size,
            compute_device=device.type,
        )

        fused_params = {
            "learning_rate": self.training_args.learning_rate,
            "weight_decay": self.training_args.weight_decay,
            "optimizer": getattr(EmbOptimType, self.training_args.optimizer.upper()),
        }
        sharders = _get_sharders(fused_params)
        
        sharding_type = tde_settings["sharding_type"]
        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=self.training_args.train_batch_size,
            constraints={
                f"{feature_name}": ParameterConstraints(
                    sharding_types=getattr(ShardingType, sharding_type.upper()).value)
                for feature_name in tde_feature_names
            }
        )
        plan: ShardingPlan = planner.collective_plan(model, sharders, pg)
        
        model = DistributedModelParallel(
            module=model,
            env=ShardingEnv.from_process_group(pg),
            device=device,
            plan=plan,
            sharders=get_default_sharders(),
        )
        attach_id_transformer_group(
            tde_settings["ps_url"],
            model, 
            tde_configs_dict)
        
        return model, tde_configs_dict, tde_feature_names, tde_settings
          
    
    def load_model(self) -> BaseRanker:
        model = self.model_class(self.cp_attr, self.model_args)
        model, tde_configs_dict, tde_feature_names, tde_settings = self._prepare_model(model)
        return model, tde_configs_dict, tde_feature_names, tde_settings

    def load_trainer(self, tde_configs_dict, tde_feature_names, tde_settings) -> TDERankerTrainer:    
        
        dense_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(self.model.named_parameters())),
            lambda _params: get_optimizer(self.training_args.optimizer, _params,
                self.training_args.learning_rate, self.training_args.weight_decay),
        )
        def get_unique_fused_optimizer(fused_optimizer, all_paths):
            unique_optimizers = []
            for path, optimizer in fused_optimizer.optimizers: 
                if path in all_paths:
                    unique_optimizers.append((path, optimizer))
            return CombinedOptimizer(unique_optimizers)
        
        unique_fused_optimizer = get_unique_fused_optimizer(
            self.model.fused_optimizer, 
            set(list(zip(*list(self.model.module.named_modules())))[0])
        ) # to make fused optimizer able to be prepared 

        self.optimizer = CombinedOptimizer([unique_fused_optimizer, dense_optimizer])   
        
        self.lr_scheduler = get_lr_scheduler()
        # self.training_args.dataloader_num_workers = 0   # avoid multi-processing

        trainer = TDERankerTrainer(
            model=self.model,
            tde_configs_dict=tde_configs_dict,
            tde_feature_names=tde_feature_names,
            tde_settings=tde_settings,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            optimizers=[self.optimizer, self.lr_scheduler],
            
        )
        
        return trainer
