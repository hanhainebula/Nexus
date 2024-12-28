# scripts to convert onnx to tensorrt

ONNX_PATH='/data2/OpenLLMs/bge-reranker-base/onnx/model.onnx' # onnx model path
TRT_SAVE_PATH='/data2/OpenLLMs/bge-reranker-base/trt/model.trt' # tensorrt model path, dirpath should be created early


TRT_PATH='/data2/home/angqing/tensorrt/TensorRT-10.7.0.23' # your tensorrt path here

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_PATH/lib
export PATH=$PATH:$TRT_PATH/bin


# Convert ONNX to TensorRT with dynamic shapes
trtexec --onnx=$ONNX_PATH --saveEngine=$TRT_SAVE_PATH --minShapes=input_ids:1x1,attention_mask:1x1 --optShapes=input_ids:1x128,attention_mask:1x128 --maxShapes=input_ids:32x512,attention_mask:32x512