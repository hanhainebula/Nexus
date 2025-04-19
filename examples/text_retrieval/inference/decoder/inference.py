import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import time
import json
from tqdm import tqdm
from Nexus import TextEmbedder, AbsInferenceArguments, BaseLLMEmbedder

model_path='/share/project/aqjiang/models/gte-Qwen2-1_5B-Instruct'

embedder = BaseLLMEmbedder(
    model_name_or_path=model_path,
    trust_remote_code=True,
    batch_size=16
)


sentence = ['你好你好']*10

embeddings = embedder.encode_query(sentence)
import pdb
pdb.set_trace()
