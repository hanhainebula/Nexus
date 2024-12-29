# scripts to convert onnx to tensorrt

ONNX_PATH='/data1/home/recstudio/haoran/UniRetrieval/recommender_results/mlp_ranker/model_onnx.pb' # onnx model path
TRT_SAVE_PATH='/data1/home/recstudio/haoran/UniRetrieval/recommender_results/mlp_ranker/model_trt.engine' # tensorrt model path, dirpath should be created early

# your tensorrt path here
TRT_PATH='/data1/home/recstudio/haoran/tensorrt/TensorRT-10.6.0.26'

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_PATH/lib
export PATH=$PATH:$TRT_PATH/bin


# Convert ONNX to TensorRT with dynamic shapes
trtexec --onnx=$ONNX_PATH --saveEngine=$TRT_SAVE_PATH --verbose