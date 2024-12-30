import yaml
import argparse
import pandas as pd
from UniRetrieval.inference.embedder.recommendation import BaseEmbedderInferenceEngine
import pycuda.driver as cuda



if __name__ == '__main__':
    infer_config_path = "/data1/home/recstudio/haoran/UniRetrieval/recommender_examples/inference/config/recflow_infer_retrieval_config.yaml"

    with open(infer_config_path, 'r') as f:
        config = yaml.safe_load(f)

    
    retriever_inference_engine = BaseEmbedderInferenceEngine(config)
    # if retriever_inference_engine.config['infer_mode'] == 'trt':
    #     cuda.Context.pop()
        
    infer_df = pd.read_feather('/data1/home/recstudio/haoran/RecStudio-Industry/inference/inference_data/recflow/recflow_infer_data.feather')
    for batch_idx in range(10):
        print(f"This is batch {batch_idx}")
        batch_st = batch_idx * 128 
        batch_ed = (batch_idx + 1) * 128 
        batch_infer_df = infer_df.iloc[batch_st:batch_ed]
        retriever_outputs = retriever_inference_engine.batch_inference(batch_infer_df)
        print(type(retriever_outputs), retriever_outputs.shape)
    if retriever_inference_engine.config['infer_mode'] == 'trt':
        cuda.Context.pop()