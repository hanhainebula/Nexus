import yaml

import numpy as np
import pandas as pd

from Nexus.inference.reranker.recommendation import BaseRerankerInferenceEngine

if __name__ == '__main__':
    infer_config_path = "./infer_config.yaml"

    with open(infer_config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
    
    rank_inference_engine = BaseRerankerInferenceEngine(config)
    
    infer_df = pd.read_feather('./recflow_infer_data.feather')
    
    item_df = pd.read_feather('./realshow_test_video_info.feather')
    
    all_item_ids = np.array(item_df['video_id'])
    
    total_batch = 10
    batch_size = 128
    rerank_candidate_num  =50
    for batch_idx in range(total_batch):
        print(f"This is batch {batch_idx}")
        
        batch_st = batch_idx * batch_size 
        batch_ed = (batch_idx + 1) * batch_size 
        
        batch_infer_df = infer_df.iloc[batch_st:batch_ed]
        
        np.random.seed(42)
        
        batch_candidates = np.random.choice(all_item_ids, size=(batch_size, rerank_candidate_num))
        
        batch_candidates_df = pd.DataFrame({rank_inference_engine.feature_config['fiid']: batch_candidates.tolist()})
        
        ranker_outputs = rank_inference_engine.batch_inference(batch_infer_df, batch_candidates_df)
        
        print(type(ranker_outputs), ranker_outputs.shape, ranker_outputs[-5:])
        
    if rank_inference_engine.config['infer_mode'] == 'trt':
        rank_inference_engine.context.pop()



