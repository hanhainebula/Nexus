import os

os.environ['CUDA_VISIBLE_DEVICES']='7'

import time
import json
from tqdm import tqdm
from Nexus import TextEmbedder, AbsInferenceArguments, BaseEmbedderInferenceEngine
# normal
model_path='/root/models/bge-base-zh-v1.5'
dataset_path = '/root/datasets/msmarco_hn_train.jsonl'
sentences = []

# with open(dataset_path,'r', encoding='utf-8') as f:
#     for line in tqdm(f):
#         obj=json.loads(line)
#         sentences.extend(obj['pos'])
#         sentences.extend(obj['neg'])

# print(len(sentences))      
# sentences = sentences[:100000]  

# with open('/root/datasets/test_inference.json','w',encoding='utf-8') as f:
#     json.dump(sentences,f, ensure_ascii=False)
with open('/root/datasets/test_inference.json','r', encoding='utf-8') as f:
    sentences = json.load(f)
         


# args=AbsInferenceArguments(
#     model_name_or_path=model_path,
#     infer_mode='normal',
#     infer_device=0,
#     infer_batch_size=16
# )
# args.infer_mode='normal'
# inference_engine=BaseEmbedderInferenceEngine(args)

# start = time.time()
# emb_normal=inference_engine.inference(sentences, batch_size=16, normalize=True)
# end = time.time()
# print(f'normal: {end - start}') # normal: 63.63837957382202

# onnx
onnx_model_path='/root/models/bge-base-zh-v1.5/onnx/model_fp16.onnx'

# args=AbsInferenceArguments(
#     model_name_or_path=model_path,
#     onnx_model_path=onnx_model_path,
#     trt_model_path=None,
#     infer_mode='onnx',
#     infer_device=0,
#     infer_batch_size=16
# )
# inference_engine_onnx = BaseEmbedderInferenceEngine(args)

# start = time.time()
# emb_onnx = inference_engine_onnx._inference_onnx(sentences, normalize=True, batch_size=512)
# end = time.time()
# print(f'onnx: {end-start}')
# tensorrt

trt_model_path ='/root/models/bge-base-zh-v1.5/trt/model_fp16.trt'


args=AbsInferenceArguments(
    model_name_or_path=model_path,
    onnx_model_path=onnx_model_path,
    trt_model_path=trt_model_path,
    infer_mode='tensorrt',
    infer_device=0,
    infer_batch_size=32
)


inference_engine_tensorrt=BaseEmbedderInferenceEngine(args)

start=time.time()
emb_trt=inference_engine_tensorrt.inference(sentences, normalize=True, batch_size=64)
end=time.time()
print(f'tensorrt: {end - start}')