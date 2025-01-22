import os 
import netifaces
from Nexus.training.embedder.recommendation.tde_runner import TDERetrieverRunner
from Nexus.training.embedder.recommendation.modeling import MLPRetriever

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
    if hasattr(runner.model, "_id_transformer_group"):
        runner.model._id_transformer_group.__del__()

if __name__ == "__main__":
    local_interfaces = get_local_interfaces()
    # set the interface for Gloo, refer to issue https://github.com/pytorch/pytorch/issues/68726#issuecomment-1813807190
    os.environ["TP_SOCKET_IFNAME"] = local_interfaces[0]
    os.environ["GLOO_SOCKET_IFNAME"] = local_interfaces[0]
    main()
