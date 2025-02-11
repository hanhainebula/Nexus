from Nexus.training.embedder.recommendation.runner import RetrieverRunner
from Nexus.training.embedder.recommendation.modeling import MLPRetriever, BaseRetriever
from Nexus.modules.query_encoder import MLPQueryEncoder
from Nexus.modules.item_encoder import MLPItemEncoder
from Nexus.modules.sampler import UniformSampler, RetrieverSampler, MaskedUniformSampler, PopularSampler, MIDXUniformSampler, MIDXPopSampler, ClusterUniformSampler, ClusterPopSampler, LSHSampler
from Nexus.modules.score import InnerProductScorer
from Nexus.modules.loss import BPRLoss
import os
import numpy as np
import torch
class MYMLPRetriever(BaseRetriever):
    def __init__(self, retriever_data_config, retriever_model_config, item_loader=None, *args, **kwargs):
        super().__init__(data_config=retriever_data_config, model_config=retriever_model_config, item_loader=item_loader, *args, **kwargs)

    def get_item_encoder(self):
        return MLPItemEncoder(self.data_config, self.model_config)
    
    def get_query_encoder(self):
        return MLPQueryEncoder(self.data_config, self.model_config, self.item_encoder)
    
    def get_score_function(self):
        return InnerProductScorer()
    
    def get_loss_function(self):
        return BPRLoss()
    
    def get_negative_sampler(self):
        return MaskedUniformSampler(num_items=self.data_config.num_items)
        
        # model_config_path = "./examples/recommendation/eval/eval_model_config.json"
        # model_args = RecommenderEvalModelArgs.from_json(model_config_path)
        # retriever:BaseRetriever = BaseRetriever.from_pretrained(model_args.retriever_ckpt_path, model_cls=MLPRetriever)
        # print("start loading retriever")
        # checkpoint = torch.load(os.path.join(model_args.retriever_ckpt_path, 'model.pt'), weights_only=True)
        # retriever.load_state_dict(checkpoint)
        # retriever.eval()
        # print("loaded retriever")
        # return RetrieverSampler(num_items=self.data_config.num_items, retriever=retriever, item_loader=self.item_loader)
        

        # return PopularSampler(pop_count=np.array([1 for _ in range(self.data_config.num_items)]))
        
        # sampler = MIDXUniformSampler(num_items=self.data_config.num_items, num_clusters=5, scorer_fn=CosineScorer)
        # sampler.update(item_embs=torch.ones((self.data_config.num_items, 8)), max_iter=2)
        # return sampler
        
        # sampler = MIDXPopSampler(pop_count=torch.tensor([1 for _ in range(self.data_config.num_items)]), num_clusters=5, mode=1)
        # sampler.update(item_embs=torch.ones((self.data_config.num_items, 8)), max_iter=2)
        # return sampler
        
        # sampler = ClusterUniformSampler(num_items=self.data_config.num_items, num_clusters=5)
        # sampler.update(item_embs=torch.ones((self.data_config.num_items, 8)), max_iter=2)
        # return sampler
        
        # sampler = ClusterPopSampler(pop_count=torch.tensor([1 for _ in range(self.data_config.num_items)]), num_clusters=10)
        # sampler.update(item_embs=torch.ones((self.data_config.num_items, 8)), max_iter=2)
        # return sampler
        
        # sampler = LSHSampler(num_items=self.data_config.num_items, n_dims=8)
        # sampler.update(item_embs=torch.ones((self.data_config.num_items, 8)))
        # return sampler
    
    def encode_info(self, *args, **kwargs):
        return super().encode_info(*args, **kwargs)
    
        
def main():
    data_config_path = "./examples/recommendation/config/data/recflow_retriever.json"
    train_config_path = "./examples/recommendation/config/mlp_retriever/train.json"
    model_config_path = "./examples/recommendation/config/mlp_retriever/model.json"
    
    runner = RetrieverRunner(
        model_config_or_path=model_config_path,
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=MLPRetriever,
    )
    runner.run()
    
    


if __name__ == "__main__":
    main()
