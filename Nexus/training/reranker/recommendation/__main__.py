import torch.distributed as dist

dist.init_process_group("nccl")  # 确保分布式环境已初始化
world_size = dist.get_world_size()
print(f"World size: {world_size}")


from Nexus.training.reranker.recommendation.runner import RankerRunner
from Nexus.training.reranker.recommendation.modeling import MLPRanker


def main():
    data_config_path = "./examples/recommendation/config/data/recflow_ranker.json"
    train_config_path = "./examples/recommendation/config/mlp_ranker/train.json"
    model_config_path = "./examples/recommendation/config/mlp_ranker/model.json"
    
    runner = RankerRunner(
        model_config_or_path=model_config_path,
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=MLPRanker
    )
    runner.run()


if __name__ == "__main__":
    main()
