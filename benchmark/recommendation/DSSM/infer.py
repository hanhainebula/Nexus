import yaml
from Nexus.inference.embedder.recommendation import BaseEmbedderInferenceEngine

import pandas as pd


if __name__ == '__main__':
    infer_config_path = "./infer_config.yaml"

    with open(infer_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    retriever_inference_engine = BaseEmbedderInferenceEngine(config)
        
    infer_df = pd.read_feather('./recflow_infer_data.feather')
    
    total_batch = 10
    batch_size = 128
    
    for batch_idx in range(total_batch):
        print(f"This is batch {batch_idx}")
        
        batch_st = batch_idx * batch_size 
        batch_ed = (batch_idx + 1) * batch_size 
        
        batch_infer_df = infer_df.iloc[batch_st:batch_ed]
        
        retriever_outputs = retriever_inference_engine.batch_inference(batch_infer_df)
        
        print(type(retriever_outputs), retriever_outputs.shape, retriever_outputs)
        
    if retriever_inference_engine.config['infer_mode'] == 'trt':
        retriever_inference_engine.context.pop()