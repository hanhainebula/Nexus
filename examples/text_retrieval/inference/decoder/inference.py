# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'

import time
import json
from tqdm import tqdm
from Nexus import TextEmbedder, AbsInferenceArguments, BaseLLMEmbedder, BaseLLMEMbedderInferenceEngine, AbsLLMInferenceArguments


    



if __name__=='__main__':
    data_path = '/share/project/nexus/datasets/test_datasets/fiqa/corpus.jsonl'
    datas = []
    with open(data_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            datas.append(obj['text'])
            if len(datas)==1000:
                break
        
    
    embedder = BaseLLMEmbedder(
    model_name_or_path='BAAI/bge-multilingual-gemma2',
    trust_remote_code=True,
    batch_size=16
    )
    norm_start = time.time()
    embeddings = embedder.encode_query(datas)
    norm_end = time.time()
    

    

    config = AbsLLMInferenceArguments(
        model_name_or_path='BAAI/bge-multilingual-gemma2',
        infer_batch_size=16,
        tensor_parallel_size=2
    )
    embedder = BaseLLMEMbedderInferenceEngine(config)
    
    start = time.time()
    embeddings = embedder.encode_query(datas)
    end = time.time()


    print('========================')
    print('normal cost:', norm_end-norm_start)
    print('========================')
    print('vllm cost:', end-start)
    print('========================')
    
    