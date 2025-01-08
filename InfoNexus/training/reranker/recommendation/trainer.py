import os
from loguru import logger

from InfoNexus.abc.training.reranker import AbsRerankerTrainer

import torch
import torch.optim as optim

from torchrec.distributed.model_parallel import DistributedModelParallel
from dynamic_embedding.wrappers import attach_id_transformer_group, wrap_dataloader, wrap_dataset
from torchrec_dynamic_embedding.dataloader import save as tde_save 


class RankerTrainer(AbsRerankerTrainer):
    def __init__(self, model, train=True,  
                 *args, **kwargs):
        super(RankerTrainer, self).__init__(model, *args, **kwargs)
        self.train_mode = train
        self.model_type = model.model_type
        if self.accelerator.is_main_process:
            print(model)
        
    def compute_loss(self, model, batch, return_outputs=False,*args, **kwargs):
        outputs = model(batch=batch, cal_loss=True,*args, **kwargs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
    def save_model(self, output_dir = None, **kwargs):
        """ Save the best model. """
        if self.accelerator.is_main_process:
            checkpoint_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save(checkpoint_dir)
            logger.info(f"Model saved in {checkpoint_dir}.")
            
            
class TDERankerTrainer(AbsRerankerTrainer):
    def __init__(self, model:DistributedModelParallel, tde_configs_dict, tde_feature_names, tde_settings, train=True, *args, **kwargs):
        super(TDERankerTrainer, self).__init__(model, *args, **kwargs)
        self.model:DistributedModelParallel = self.accelerator.unwrap_model(self.model)
        self.train_mode = train
        self.model_type = model.module.model_type
        if self.accelerator.is_main_process:
            print(model)
        self.tde_configs_dict = tde_configs_dict
        self.tde_feature_names = tde_feature_names
        self.tde_settings = tde_settings
            
    def get_train_dataloader(self):
        train_dataloader = super().get_train_dataloader()
        return wrap_dataloader(dataloader=train_dataloader, module=self.model, configs_dict=self.tde_configs_dict)
        
    def compute_loss(self, model, batch, return_outputs=False,*args, **kwargs):
        model = self.accelerator.unwrap_model(model)
        
        outputs = model(batch=batch, cal_loss=True,*args, **kwargs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
    def save_model(self, output_dir = None, **kwargs):
        
        """ Save the best model. """
        if self.accelerator.is_main_process:
            checkpoint_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.model.module.save(checkpoint_dir)
            logger.info(f"Model saved in {checkpoint_dir}.")
            tde_save(self.model)
        

