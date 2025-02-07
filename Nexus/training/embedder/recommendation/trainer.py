import os
from Nexus.abc.training.embedder import AbsEmbedderTrainer
import torch
from torch.utils.data import DataLoader
from loguru import logger

from torchrec.distributed.model_parallel import DistributedModelParallel
from dynamic_embedding.wrappers import wrap_dataloader, wrap_dataset
from torchrec_dynamic_embedding.dataloader import save as tde_save 
from tqdm import tqdm

class RetrieverTrainer(AbsEmbedderTrainer):
    def __init__(self, model, train=True, *args, **kwargs):
        super(RetrieverTrainer, self).__init__(model, *args, **kwargs)
        self.train_mode = train
        self.model_type = model.model_type
        # self._check_checkpoint_dir()
        if self.accelerator.is_main_process:
            print(model)
        
        # self.item_vectors = None
        # self.item_ids = None
        
    def compute_loss(self, model, batch, return_outputs=False,*args, **kwargs):
        outputs = model(batch=batch, cal_loss=True,*args, **kwargs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
        
    def save_model(self, output_dir = None, **kwargs):
        """ Save the best model. """
        item_vectors, item_ids = self.update_item_vectors(output_dir)
        if self.accelerator.is_main_process:
            checkpoint_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save(checkpoint_dir)
            logger.info(f"Model saved in {checkpoint_dir}.")         
            item_vectors_path = os.path.join(checkpoint_dir, 'item_vectors.pt')
            torch.save({'item_vectors': item_vectors, 'item_ids': item_ids}, item_vectors_path)
            logger.info(f'Item vectors saved to {checkpoint_dir}.')
            
    
    def update_item_vectors(self, output_dir):
        with torch.no_grad():
            logger.info(f'Update item vectors...')
            self.model.eval()
            all_item_vectors, all_item_ids = [], []
            model = self.accelerator.unwrap_model(self.model)
            for item_batch in self.accelerator.prepare(model.item_loader):
                item_vector = model.item_encoder(item_batch)
                all_item_vectors.append(item_vector)
                all_item_ids.append(item_batch[model.fiid])
            all_item_vectors = self.accelerator.gather_for_metrics(all_item_vectors)
            all_item_ids = self.accelerator.gather_for_metrics(all_item_ids)
            all_item_vectors = torch.cat(all_item_vectors, dim=0)
            all_item_ids = torch.cat(all_item_ids, dim=0).cpu()
            return all_item_vectors, all_item_ids
    

class TDERetrieverTrainer(AbsEmbedderTrainer):
    def __init__(self, model, tde_configs_dict, tde_feature_names, tde_settings, train=True, *args, **kwargs):
        super(TDERetrieverTrainer, self).__init__(model, *args, **kwargs)
        self.model:DistributedModelParallel = self.accelerator.unwrap_model(self.model)
        self.train_mode = train
        self.model_type = self.model.module.model_type
        # self._check_checkpoint_dir()
        if self.accelerator.is_main_process:
            print(model)
        self.tde_configs_dict = tde_configs_dict
        self.tde_feature_names = tde_feature_names
        self.tde_settings = tde_settings

        item_dataset = wrap_dataset(
            dataset=self.model.module.item_loader.dataset,
            module=self.model,
            configs_dict=self.tde_configs_dict
        )
        
        # attach item_loader to base_model, so that we can use it in base_model 
        self.model.module.base_model.item_loader = DataLoader(item_dataset, 
                                                              batch_size=self.model.module.item_loader.batch_size,
                                                              num_workers=self.model.module.item_loader.num_workers, 
                                                              shuffle=False)
        # Do not prepare item_loader
        # self.model.module.base_model.item_loader = self.accelerator.prepare(self.model.module.base_model.item_loader)
        self.model.module.base_model.item_loader = wrap_dataloader(
            self.model.module.base_model.item_loader, self.model, self.tde_configs_dict)
        # self.item_vectors = None
        # self.item_ids = None
        
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
        # call save method on every rank
        item_vectors, item_ids = self.update_item_vectors(output_dir)
        
        checkpoint_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.module.save(checkpoint_dir)
        if self.accelerator.is_main_process:
            logger.info(f"Model saved in {checkpoint_dir}.")
        
        if self.accelerator.is_main_process:
            item_vectors_path = os.path.join(checkpoint_dir, 'item_vectors.pt')
            torch.save({'item_vectors': item_vectors, 'item_ids': item_ids}, item_vectors_path)
            logger.info(f'Item vectors saved to {checkpoint_dir}.')
        
        tde_save(self.model)
        if self.accelerator.is_main_process:
            logger.info(f"IDTransformer saved.")
    
    def _batch_to_device(self, batch_data:dict, device):
        '''
        move batch data to device
        Args:
            batch_data: dict
        Returns:
            batch_data: dict
        '''
        for key, value in batch_data.items():
            if isinstance(value, dict):
                batch_data[key] = self._batch_to_device(value, device)
            else:
                batch_data[key] = value.to(device)
        return batch_data
    
    def update_item_vectors(self, output_dir):
        with torch.no_grad():
            logger.info(f'Update item vectors...')
            self.model.eval()
            all_item_vectors = []
            for item_batch in tqdm(self.model.module.item_loader):
                item_batch = self._batch_to_device(item_batch, self.model.device)
                item_vector = self.model.module.item_encoder(item_batch)
                all_item_vectors.append(item_vector.detach().cpu())
            # Don't use data parallel here, as item_vectors should be placed on cpu
            # TODO: make sure item vectors are gathered in the same order as in dataset 
            all_item_vectors = torch.cat(all_item_vectors, dim=0)
            all_item_ids = torch.from_numpy(self.model.module.item_loader.dataset._dataset.item_ids)
            return all_item_vectors, all_item_ids
