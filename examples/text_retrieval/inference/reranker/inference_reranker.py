import os
os.environ['CUDA_VISIBLE_DEVICES']='5'

import time
import json
from tqdm import tqdm
from Nexus import TextEmbedder, AbsInferenceArguments, BaseRerankerInferenceEngine

model_path='/root/models/bge-reranker-base'
onnx_model_path='/root/models/bge-reranker-base/onnx/model_fp16.onnx'
trt_model_path ='/root/models/bge-reranker-base/trt/model_fp16.trt'



args=AbsInferenceArguments(
    model_name_or_path=model_path,
    onnx_model_path=onnx_model_path,
    trt_model_path=trt_model_path,
    infer_mode='normal',
    infer_device=0,
    infer_batch_size=32
)

######################## Test inference accuracy ########################
qa_pairs = [['你好你好','你好，你的名字是？']]

"""
1. Test normal
"""
args.infer_mode='normal'
inference_engine=BaseRerankerInferenceEngine(args)

emb_normal=inference_engine.inference(qa_pairs, normalize=False)
print(emb_normal)
del inference_engine

"""
2. Test onnx
"""
args.infer_mode='onnx'
# BaseRerankerInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=onnx_model_path, use_fp16=True, opset_version=17)
inference_engine_onnx = BaseRerankerInferenceEngine(args)

emb_onnx = inference_engine_onnx.inference(qa_pairs, normalize=False)
print(emb_onnx)
del inference_engine_onnx

"""
3. Test tensorrt
"""
args.infer_mode='tensorrt'
inference_engine_tensorrt=BaseRerankerInferenceEngine(args)

emb_trt=inference_engine_tensorrt.inference(qa_pairs, normalize=False)
print(emb_trt)
del inference_engine_tensorrt

######################## Test inference speed ########################
with open('/root/datasets/test_inference_reranker.json', 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)
    
"""
1. Test normal
"""
args.infer_mode='normal'
inference_engine=BaseRerankerInferenceEngine(args)

start = time.time()
emb_normal=inference_engine.inference(qa_pairs, batch_size=256, normalize=True)
end = time.time()
print(f'normal: {end - start}') 
del inference_engine

"""
2. Test onnx
"""
args.infer_mode='onnx'
# BaseRerankerInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=onnx_model_path, use_fp16=True, opset_version=17)
inference_engine_onnx = BaseRerankerInferenceEngine(args)

start = time.time()
emb_onnx = inference_engine_onnx._inference_onnx(qa_pairs, normalize=False, batch_size=128)
end = time.time()
print(f'onnx: {end-start}') # onnx: 191.85031652450562
del inference_engine_onnx

"""
3. Test tensorrt
"""
args.infer_mode='tensorrt'
inference_engine_tensorrt=BaseRerankerInferenceEngine(args)

start=time.time()
emb_trt=inference_engine_tensorrt.inference(qa_pairs, normalize=False, batch_size=256)
end=time.time()
print(f'tensorrt: {end - start}') # tensorrt: 590.0577092170715
del inference_engine_tensorrt

# """
# bs=256
# qa = qa*3
# normal: 312.16792941093445
# onnx: 191.85031652450562
# tensorrt: 590.0577092170715
# """

"""
reranker
bs=256
normal: 255.43481874465942
onnx: 65.45748233795166
tensorrt: 176.57225012779236
"""

"""
bs=512
normal: 
onnx: 74.49612545967102
tensorrt: 
"""

"""
bs=128
normal: 
onnx: 1min
tensorrt: 3min
"""

