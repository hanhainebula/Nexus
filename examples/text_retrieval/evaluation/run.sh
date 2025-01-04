BASE_DIR=/data1/home/recstudio/angqing/UniRetrieval
EMBEDDER=/data1/home/recstudio/angqing/models/bge-base-zh-v1.5
RERANKER=/data1/home/recstudio/angqing/models/bge-reranker-base
embedder_infer_mode=onnx
reranker_infer_mode=onnx
embedder_onnx_path=
reranker_onnx_path=
embedder_trt_path=
reranker_trt_path=


cd $BASE_DIR


python -m UniRetrieval.evaluation.text_retrieval.airbench \
    --benchmark_version AIR-Bench_24.05 \
    --task_types qa \
    --domains arxiv \
    --languages en \
    --splits dev test \
    --output_dir $BASE_DIR/examples/text_retrieval/evaluation/air_bench/search_results \
    --search_top_k 1000 \
    --rerank_top_k 20 \
    --cache_dir $BASE_DIR/examples/text_retrieval/evaluation/cache/data \
    --overwrite False \
    --embedder_name_or_path $EMBEDDER \
    --reranker_name_or_path $RERANKER \
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