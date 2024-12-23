from transformers import HfArgumentParser

from UniRetrieval.training.reranker.recommendation import *

from UniRetrieval.training.reranker.recommendation.datasets import get_datasets

from UniRetrieval.training.reranker.recommendation.trainer import RankerTrainer

import yaml



def main():
    ranker_data_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/data/recflow_ranker.json"
    ranker_train_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/mlp_ranker/train.json"
    ranker_model_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/mlp_ranker/model.json"

    (ranker_train_data, ranker_eval_data), ranker_data_config = get_datasets(ranker_data_config_path)

    ranker_model = MLPRanker(ranker_data_config, ranker_model_config_path)

    ranker_trainer = RankerTrainer(ranker_model, ranker_train_config_path)

    ranker_trainer.train(ranker_train_data, ranker_eval_data)


if __name__ == "__main__":
    main()
