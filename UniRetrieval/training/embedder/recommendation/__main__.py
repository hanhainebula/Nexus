from transformers import HfArgumentParser

from UniRetrieval.training.embedder.recommendation import *

from UniRetrieval.training.embedder.recommendation.datasets import get_datasets

from UniRetrieval.training.embedder.recommendation.trainer import RetrieverTrainer

import yaml



def main():
    retriever_data_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/data/recflow_retriever.json"
    retriever_train_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/mlp_retriever/train.json"
    retriever_model_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/config/mlp_retriever/model.json"

    (retriever_train_data, retriever_eval_data), retriever_data_config = get_datasets(retriever_data_config_path)

    retriever_model = MLPRetriever(retriever_data_config, retriever_model_config_path)

    retriever_trainer = RetrieverTrainer(
        model=retriever_model, 
        config=retriever_train_config_path
    )

    retriever_trainer.train(retriever_train_data, retriever_eval_data)


if __name__ == "__main__":
    main()
