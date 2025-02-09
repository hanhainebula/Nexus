export HF_ENDPOINT="https://hf-mirror.com"
BASE_DIR=/data1/home/recstudio/angqing/InfoNexus
DATASET_DIR=/root/datasets/test_datasets/fiqa
EMBEDDER=/data1/home/recstudio/angqing/models/bge-base-zh-v1.5
RERANKER=/data1/home/recstudio/angqing/models/bge-reranker-base
embedder_infer_mode=normal
reranker_infer_mode=normal
embedder_onnx_path=$EMBEDDER/onnx/model_fp16.onnx
reranker_onnx_path=$RERANKER/onnx/model_fp16.onnx
embedder_trt_path=$EMBEDDER/trt/model_fp16.trt
reranker_trt_path=$RERANKER/trt/model_fp16.trt


cd $BASE_DIR


python -m Nexus.evaluation.text_retrieval \
    --dataset_dir $DATASET_DIR \
    --dataset_names fiqa \
    --eval_name fiqa_eval \
    --splits test \
    --eval_output_dir $BASE_DIR/examples/text_retrieval/evaluation/fiqa/search_results \
    --search_top_k 1000 \
    --rerank_top_k 20 \
    --cache_dir $BASE_DIR/examples/text_retrieval/evaluation/cache/data \
    --overwrite False \
    --embedder_name_or_path $EMBEDDER \
    --reranker_name_or_path $RERANKER \
    --embedder_batch_size 32 \
    --reranker_batch_size 8 \
    --devices cuda:0 \
    --model_cache_dir $BASE_DIR/examples/text_retrieval/evaluation/cache/model \
    --reranker_query_max_length 128 \
    --reranker_max_length 512 \
    --embedder_infer_mode $embedder_infer_mode \
    --reranker_infer_mode $reranker_infer_mode \
    --embedder_onnx_model_path $embedder_onnx_path \
    --reranker_onnx_model_path $reranker_onnx_path \
    --embedder_trt_model_path $embedder_trt_path \
    --reranker_trt_model_path $reranker_trt_path