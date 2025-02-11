import os
import netifaces

from dataclasses import dataclass, field

from Nexus.training.reranker.recommendation.modeling import DCNv2Ranker
from Nexus.training.reranker.recommendation.arguments import ModelArguments
from Nexus.training.reranker.recommendation.tde_runner import TDERankerRunner

def get_local_interfaces():
    local_interfaces = []
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        is_lan_interface = False
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                ip = addr['addr']
                if ip.startswith("172."):  # only consider LAN IP addresses
                    is_lan_interface = True
        if is_lan_interface:
            local_interfaces.append(interface)
    return local_interfaces

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
    train_config_path = "./tde_training_config.json"
    model_config_path = "./tde_model_config.json"
    
    runner = TDERankerRunner(
        model_config_or_path=DCNv2ModelArguments.from_json(model_config_path),
        data_config_or_path=data_config_path,
        train_config_or_path=train_config_path,
        model_class=DCNv2Ranker
    )
    runner.run()
    
    if hasattr(runner.model, "_id_transformer_group"):
        runner.model._id_transformer_group.__del__()


if __name__ == "__main__":
    local_interfaces = get_local_interfaces()
    os.environ["TP_SOCKET_IFNAME"] = local_interfaces[0]
    os.environ["GLOO_SOCKET_IFNAME"] = local_interfaces[0]
    main()