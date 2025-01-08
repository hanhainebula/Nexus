import yaml
import argparse
import pandas as pd
from Nexus.inference.reranker.recommendation import BaseRerankerInferenceEngine
import pycuda.driver as cuda
import numpy as np

if __name__ == '__main__':
    # infer_config_path = "/data1/home/recstudio/haoran/angqing_temp/mlp_reranker/recflow_infer_ranker_config.yaml"
    infer_config_path = "./examples/recommendation/inference/config/recflow_infer_ranker_config.yaml"
    
    with open(infer_config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(config)

    rank_inference_engine = BaseRerankerInferenceEngine(config)
        
    infer_df = pd.read_feather('/data1/home/recstudio/haoran/InfoNexus/examples/recommendation/inference/inference_data/recflow/recflow_infer_data.feather')
    item_df = pd.read_feather('/data1/home/recstudio/haoran/InfoNexus/examples/recommendation/inference/inference_data/recflow/realshow_test_video_info.feather')
    all_item_ids = np.array(item_df['video_id'])
    for batch_idx in range(10):
        print(f"This is batch {batch_idx}")
        batch_st = batch_idx * 128 
        batch_ed = (batch_idx + 1) * 128 
        batch_infer_df = infer_df.iloc[batch_st:batch_ed]
        np.random.seed(42)
        batch_candidates = np.random.choice(all_item_ids, size=(128, 50))
        batch_candidates_df = pd.DataFrame({rank_inference_engine.feature_config['fiid']: batch_candidates.tolist()})
        ranker_outputs = rank_inference_engine.batch_inference(batch_infer_df, batch_candidates_df)
        print(type(ranker_outputs), ranker_outputs.shape, ranker_outputs[-5:])
        
    if rank_inference_engine.config['infer_mode'] == 'trt':
        cuda.Context.pop()