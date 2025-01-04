# scripts to convert onnx to tensorrt

ONNX_PATH='' # onnx model path
TRT_SAVE_PATH='' # tensorrt model path, dirpath should be created early

# your tensorrt path here
TRT_PATH=''

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_PATH/lib
export PATH=$PATH:$TRT_PATH/bin


# Convert ONNX to TensorRT with dynamic shapes
# embedder:
trtexec --onnx=$ONNX_PATH --saveEngine=$TRT_SAVE_PATH --minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 --optShapes=input_ids:8x128,attention_mask:8x128,token_type_ids:8x128 --maxShapes=input_ids:16x512,attention_mask:16x512,token_type_ids:16x512 --verbose

# reranker:
# trtexec --onnx=$ONNX_PATH --saveEngine=$TRT_SAVE_PATH --minShapes=input_ids:1x1,attention_mask:1x1 --optShapes=input_ids:8x128,attention_mask:8x128 --maxShapes=input_ids:16x512,attention_mask:16x512