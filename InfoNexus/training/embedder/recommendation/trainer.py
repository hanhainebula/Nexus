import os
from InfoNexus.abc.training.embedder import AbsEmbedderTrainer
import torch
from loguru import logger

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
            # self.trainer.item_vectors = all_item_vectors
            # self.trainer.item_ids = all_item_ids
            
            # logger.info(f'Item vectors updated.')
            # if self.accelerator.is_main_process:
            #     if model.model_type == "retriever":
            #         item_vectors_path = os.path.join(output_dir, 'item_vectors.pt')
            #         torch.save({'item_vectors': all_item_vectors, 'item_ids': all_item_ids}, item_vectors_path)
            #         logger.info(f'Item vectors saved.')
    
        
