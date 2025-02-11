import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

import time
import json
from tqdm import tqdm
from Nexus import TextEmbedder, AbsInferenceArguments, BaseEmbedderInferenceEngine

model_path='/root/models/bge-base-zh-v1.5'
onnx_model_path='/root/models/bge-base-zh-v1.5/onnx/model_fp16.onnx'
trt_model_path ='/root/models/bge-base-zh-v1.5/trt/model_fp16.trt'

args=AbsInferenceArguments(
    model_name_or_path=model_path,
    onnx_model_path=onnx_model_path,
    trt_model_path=trt_model_path,
    infer_mode='normal',
    infer_device=0,
    infer_batch_size=32
)

################## Test inference accuracy ##################
sentences = [
"你好，你叫什么名字","你好，你叫什么名字呀？"
]
"""
1. Test Normal
"""
args.infer_mode='normal'
inference_engine=BaseEmbedderInferenceEngine(args)

emb_normal=inference_engine.inference(sentences, batch_size=512, normalize=True)
print(emb_normal[0] @ emb_normal[1].T)

del inference_engine

"""
2. Test onnx
"""
# BaseEmbedderInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=onnx_model_path, use_fp16=False, opset_version=17)
args.infer_mode='onnx'
inference_engine_onnx = BaseEmbedderInferenceEngine(args)

emb_onnx = inference_engine_onnx.inference(sentences, normalize=True, batch_size=512)
print(emb_onnx[0] @ emb_onnx[1].T)
del inference_engine_onnx

"""
3. Test tensorrt
"""
args.infer_mode='tensorrt'
inference_engine_tensorrt=BaseEmbedderInferenceEngine(args)

emb_trt=inference_engine_tensorrt.inference(sentences, normalize=True, batch_size=512)
print(emb_trt[0] @ emb_trt[1].T)
del inference_engine_tensorrt

################## Test inference speed

with open('/root/datasets/test_inference.json','r', encoding='utf-8') as f:
    sentences = json.load(f)

"""
1. Test normal
"""
args.infer_mode='normal'
inference_engine=BaseEmbedderInferenceEngine(args)

start = time.time()
emb_normal=inference_engine.inference(sentences, batch_size=512, normalize=True)
end = time.time()
print(f'normal: {end - start}') # normal: 966.1819205284119
del inference_engine

"""
2. Test onnx
"""

# BaseEmbedderInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=onnx_model_path, use_fp16=True, opset_version=17)
args.infer_mode='onnx'
inference_engine_onnx = BaseEmbedderInferenceEngine(args)

start = time.time()
emb_onnx = inference_engine_onnx.inference(sentences, normalize=False, batch_size=512)
end = time.time()
print(f'onnx: {end-start}') # onnx: 590.4739429950714
del inference_engine_onnx

"""
3. Test tensorrt
"""
args.infer_mode='tensorrt'
inference_engine_tensorrt=BaseEmbedderInferenceEngine(args)

start=time.time()
emb_trt=inference_engine_tensorrt.inference(sentences, normalize=True, batch_size=512)
end=time.time()
del inference_engine_tensorrt

# print(f'tensorrt: {end - start}')

"""
bs=512
normal: 966.1819205284119
onnx: 590.4739429950714
tensorrt: 493.34820890426636
"""