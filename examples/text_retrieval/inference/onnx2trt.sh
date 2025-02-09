# scripts to convert onnx to tensorrt

ONNX_PATH='/data1/home/recstudio/angqing/models/bge-base-zh-v1.5/onnx/model.onnx' # onnx model path
TRT_SAVE_PATH='/data1/home/recstudio/angqing/models/bge-base-zh-v1.5/trt/model.trt' # tensorrt model path, dirpath should be created early

# your tensorrt path here
TRT_PATH='/root/TensorRT-10.7.0.23'

export LD_LIBRARY_PATH=$TRT_PATH/lib:$LD_LIBRARY_PATH
export PATH=$TRT_PATH/bin:$PATH


# Convert ONNX to TensorRT with dynamic shapes
# embedder:
trtexec --onnx=$ONNX_PATH --saveEngine=$TRT_SAVE_PATH --minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 --optShapes=input_ids:8x128,attention_mask:8x128,token_type_ids:8x128 --maxShapes=input_ids:16x512,attention_mask:16x512,token_type_ids:16x512 --verbose

# reranker:
# trtexec --onnx=$ONNX_PATH --saveEngine=$TRT_SAVE_PATH --minShapes=input_ids:1x1,attention_mask:1x1 --optShapes=input_ids:8x128,attention_mask:8x128 --maxShapes=input_ids:16x512,attention_mask:16x512
