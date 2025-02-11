import yaml

import numpy as np
import pandas as pd

from Nexus.inference.reranker.recommendation import TDERerankerInferenceEngine

if __name__ == '__main__':
    infer_config_path = "./infer_config.yaml"

    with open(infer_config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
        
    with open(infer_config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
    
    rank_inference_engine = TDERerankerInferenceEngine(config)
    
    infer_df = pd.read_feather('./recflow_infer_data.feather')
    
    item_df = pd.read_feather('./realshow_test_video_info.feather')
    
    all_item_ids = np.array(item_df['video_id'])
    
    if rank_inference_engine.config['infer_mode'] == 'trt':
        rank_inference_engine.context.pop()