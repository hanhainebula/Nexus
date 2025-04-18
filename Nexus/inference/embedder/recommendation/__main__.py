import yaml
import argparse
import pandas as pd
from Nexus.inference.embedder.recommendation import BaseEmbedderInferenceEngine
import pycuda.driver as cuda

import time
from datetime import timedelta


if __name__ == '__main__':
    infer_config_path = "/data1/home/recstudio/haoran/Nexus/examples/recommendation/inference/config/recflow_infer_retrieval_config.yaml"

    with open(infer_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    retriever_inference_engine = BaseEmbedderInferenceEngine(config)
        
    infer_df = pd.read_feather('/data1/home/recstudio/haoran/Nexus/examples/recommendation/inference/inference_data/recflow/recflow_infer_data.feather')
    
    # 记录开始时间
    start_time = time.time()

    for batch_idx in range(10):
        print(f"This is batch {batch_idx}")
        batch_st = batch_idx * 128 
        batch_ed = (batch_idx + 1) * 128 
        batch_infer_df = infer_df.iloc[batch_st:batch_ed]
        retriever_outputs = retriever_inference_engine.batch_inference(batch_infer_df)
        print(type(retriever_outputs), retriever_outputs.shape, retriever_outputs)
        
    # 记录结束时间
    end_time = time.time()

    # 计算时间差
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"Total Time Cost: {elapsed_time}")


    if retriever_inference_engine.config['infer_mode'] == 'trt':
        retriever_inference_engine.context.pop()