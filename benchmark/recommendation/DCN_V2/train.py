from dataclasses import dataclass, field

from Nexus.training.reranker.recommendation.runner import RankerRunner
from Nexus.training.reranker.recommendation.modeling import DCNv2Ranker
from Nexus.training.reranker.recommendation.arguments import ModelArguments

@dataclass
class DCNv2ModelArguments(ModelArguments):
    embedding_dim: int
    cross_net_layers: int = 5
    deep_cross_combination: str = "parallel"
    mlp_layers: int = field(default=None, metadata={"nargs": "+"})
    activation: str = "relu"
    dropout: float = 0.1
    batch_norm: bool = False
    

def main():
    data_config_path = "./data_recflow_config.json"
    train_config_path = "./training_config.json"
    model_config_path = "./model_config.json"
    
    runner = RankerRunner(
        model_config_or_path=DCNv2ModelArguments.from_json(model_config_path),
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=DCNv2Ranker
    )
    runner.run()


if __name__ == "__main__":
    main()




