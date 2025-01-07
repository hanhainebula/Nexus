from UniRetrieval.training.embedder.recommendation.tde_runner import TDERetrieverRunner
from UniRetrieval.training.embedder.recommendation.modeling import MLPRetriever


def main():
    data_config_path = "./examples/recommendation/config/data/recflow_retriever.json"
    train_config_path = "./examples/recommendation/config/mlp_retriever_tde/train.json"
    model_config_path = "./examples/recommendation/config/mlp_retriever_tde/model.json"
    
    runner = TDERetrieverRunner(
        model_config_or_path=model_config_path,
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=MLPRetriever,
    )
    runner.run()


if __name__ == "__main__":
    main()
