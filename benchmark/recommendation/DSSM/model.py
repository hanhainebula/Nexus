from Nexus.training.embedder.recommendation.modeling import DSSMRetriever
from Nexus.training.embedder.recommendation.runner import RetrieverRunner


def main():
    data_config_path = "./data_recflow_config.json"
    train_config_path = "./training_config.json"
    model_config_path = "./model_config.json"
    
    runner = RetrieverRunner(
        model_config_or_path=model_config_path,
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=DSSMRetriever,
    )
    runner.run()


if __name__ == "__main__":
    main()