import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

import time
import json
from tqdm import tqdm
from Nexus import TextEmbedder, AbsInferenceArguments, BaseEmbedderInferenceEngine

model_path='/root/models/bge-base-zh-v1.5'
onnx_model_path='/root/models/bge-base-zh-v1.5/onnx/model_fp16.onnx'
trt_model_path ='/root/models/bge-base-zh-v1.5/trt/model_fp16.trt'

dataset_path = '/root/datasets/msmarco_hn_train.jsonl'

sentences=[]
with open(dataset_path,'r', encoding='utf-8') as f:
    for line in tqdm(f):
        obj=json.loads(line)
        sentences.extend(obj['pos'])
        sentences.extend(obj['neg'])

print(len(sentences))      
sentences = sentences[:500000]  

with open('/root/datasets/test_inference.json','w',encoding='utf-8') as f:
    json.dump(sentences,f, ensure_ascii=False)
with open('/root/datasets/test_inference.json','r', encoding='utf-8') as f:
    sentences = json.load(f)
         


args=AbsInferenceArguments(
    model_name_or_path=model_path,
    onnx_model_path=onnx_model_path,
    trt_model_path=trt_model_path,
    infer_mode='normal',
    infer_device=0,
    infer_batch_size=32
)

"""
1. Test normal
"""
args.infer_mode='normal'
inference_engine=BaseEmbedderInferenceEngine(args)

start = time.time()
emb_normal=inference_engine.inference(sentences, batch_size=16, normalize=True)
end = time.time()
print(f'normal: {end - start}')

"""
2. Test onnx
"""
args.infer_mode='onnx'
# BaseEmbedderInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=onnx_model_path, use_fp16=True)
inference_engine_onnx = BaseEmbedderInferenceEngine(args)

start = time.time()
emb_onnx = inference_engine_onnx._inference_onnx(sentences, normalize=True, batch_size=512)
end = time.time()
print(f'onnx: {end-start}')

"""
3. Test tensorrt
"""
args.infer_mode='tensorrt'
inference_engine_tensorrt=BaseEmbedderInferenceEngine(args)

start=time.time()
emb_trt=inference_engine_tensorrt.inference(sentences, normalize=True, batch_size=64)
end=time.time()
print(f'tensorrt: {end - start}')