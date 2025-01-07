# Examples for text_retrieval
## Train
1. Embedder
    ```bash
    export WANDB_MODE=disabled


    BASE_DIR='/data1/home/recstudio/angqing/InfoNexus'

    MODEL_NAME_OR_PATH='/data1/home/recstudio/angqing/models/bge-base-zh-v1.5'
    TRAIN_DATA="/data1/home/recstudio/angqing/InfoNexus/eval_scripts/training/text_retrieval/example_data/fiqa.jsonl"
    CKPT_SAVE_DIR='/data1/home/recstudio/angqing/InfoNexus/checkpoints'
    DEEPSPEED_DIR='/data1/home/recstudio/angqing/InfoNexus/eval_scripts/training/ds_stage0.json'
    ACCELERATE_CONFIG='/data1/home/recstudio/angqing/InfoNexus/eval_scripts/training/text_retrieval/accelerate_config_multi.json'
    # set large epochs and small batch size for testing
    num_train_epochs=2
    per_device_train_batch_size=16
    # set num_gpus to 2 for testing
    num_gpus=2

    if [ -z "$HF_HUB_CACHE" ]; then
        export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
    fi

    model_args="\
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --cache_dir $HF_HUB_CACHE \
    "

    data_args="\
        --train_data $TRAIN_DATA \
        --cache_path ~/.cache \
        --train_group_size 8 \
        --query_max_len 512 \
        --passage_max_len 512 \
        --pad_to_multiple_of 8 \
        --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
        --query_instruction_format '{}{}' \
        --knowledge_distillation False \
    "

    training_args="\
        --output_dir $CKPT_SAVE_DIR \
        --overwrite_output_dir \
        --learning_rate 1e-5 \
        --fp16 \
        --num_train_epochs $num_train_epochs \
        --per_device_train_batch_size $per_device_train_batch_size \
        --dataloader_drop_last True \
        --warmup_ratio 0.1 \
        --gradient_checkpointing \
        --deepspeed $DEEPSPEED_DIR \
        --logging_steps 10 \
        --save_steps 500 \
        --negatives_cross_device \
        --temperature 0.02 \
        --sentence_pooling_method cls \
        --normalize_embeddings True \
        --kd_loss_type kl_div \
    "

    cd $BASE_DIR

    cmd="accelerate launch --config_file $ACCELERATE_CONFIG \
        InfoNexus/training/embedder/text_retrieval/__main__.py \
        $model_args \
        $data_args \
        $training_args \
    "

    echo $cmd
    eval $cmd
    ```
2. Reranker
    ```bash
    export WANDB_MODE=disabled

    BASE_DIR='/data1/home/recstudio/angqing/InfoNexus'

    MODEL_NAME_OR_PATH='/data2/OpenLLMs/bge-reranker-base'
    TRAIN_DATA="/data1/home/recstudio/angqing/InfoNexus/eval_scripts/training/text_retrieval/example_data/fiqa.jsonl"
    CKPT_SAVE_DIR='/data2/home/angqing/code/InfoNexus/checkpoints/test_reranker'
    DEEPSPEED_DIR='/data1/home/recstudio/angqing/InfoNexus/eval_scripts/training/ds_stage0.json'
    ACCELERATE_CONFIG='/data1/home/recstudio/angqing/InfoNexus/eval_scripts/training/text_retrieval/accelerate_config_multi.json'


    # set large epochs and small batch size for testing
    num_train_epochs=2
    per_device_train_batch_size=30
    gradient_accumulation_steps=1
    train_group_size=16

    # set num_gpus to 2 for testing
    num_gpus=2

    if [ -z "$HF_HUB_CACHE" ]; then
        export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
    fi

    model_args="\
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --cache_dir $HF_HUB_CACHE \
    "

    data_args="\
        --train_data $TRAIN_DATA \
        --cache_path ~/.cache \
        --train_group_size $train_group_size \
        --query_max_len 256 \
        --passage_max_len 256 \
        --pad_to_multiple_of 8 \
        --knowledge_distillation True \
    "

    training_args="\
        --output_dir $CKPT_SAVE_DIR \
        --overwrite_output_dir \
        --learning_rate 6e-5 \
        --fp16 \
        --num_train_epochs $num_train_epochs \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --dataloader_drop_last True \
        --warmup_ratio 0.1 \
        --gradient_checkpointing \
        --weight_decay 0.01 \
        --deepspeed $DEEPSPEED_DIR \
        --logging_steps 1 \
        --save_steps 100 \
    "
    cd $BASE_DIR

    cmd="accelerate launch --config_file $ACCELERATE_CONFIG \
        InfoNexus/training/reranker/text_retrieval/__main__.py \
        $model_args \
        $data_args \
        $training_args \
    "

    echo $cmd
    eval $cmd
    ```

3. Multi-Node-training

    We Use accelerate to train models on multi-nodes.
    
    1. Generate accelerate config file.
    ```bash
    accelerate config --config_file accelerate_config.json
    ```

    2. Run above accelerate scrpits in each Node respectively.  

Detailed scripts are in ./training

## Inference
1. Embedder
    1. normal
    
        There are two ways of normal inference. The first is to use FlagModel based method, the second is to use inference engine with `infer_mode` setting to `'normal'`

        ```python
        # 1. not use inference engine
        # TODO pesudo import, should change
        from InfoNexus import TextEmbedder, AbsInferenceArguments, BaseEmbedderInferenceEngine

        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "The Eiffel Tower is located in Paris, France.",
            "Python is a popular programming language.",
            "The Great Wall of China is one of the Seven Wonders of the World.",
            "Space exploration has led to many scientific discoveries.",
            "Climate change is a pressing global issue.",
            "The Mona Lisa is a famous painting by Leonardo da Vinci.",
            "Electric cars are becoming more common.",
            "The human brain is an incredibly complex organ."
        ]

        model = TextEmbedder(model_name_or_path='/data2/OpenLLMs/bge-base-zh-v1.5', use_fp16=True, devices=['cuda:1','cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        emb_model= model.encode(sentences, batch_size = 5)

        print(emb_model.shape)
        print(emb_model[0]@ emb_model[1].T)
        # 2. using inference engine
        model_path='/data2/OpenLLMs/bge-base-zh-v1.5'
        args=AbsInferenceArguments(
            model_name_or_path=model_path,
            infer_mode='normal',
            infer_device=0,
            infer_batch_size=16
        )
        args.infer_mode='normal'
        inference_engine=BaseEmbedderInferenceEngine(args)
        emb_normal=inference_engine.inference(sentences, batch_size=10, normalize=True)
        print(emb_normal.shape)
        print(emb_normal[0]@ emb_normal[1].T)
        ```
    2. ONNX
        
        Convert pytorch model to onnx first.

        ```python
        from InfoNexus import AbsInferenceArguments, BaseEmbedderInferenceEngine
        model_path='/data2/OpenLLMs/bge-base-zh-v1.5'
        onnx_model_path='/data2/OpenLLMs/bge-base-zh-v1.5/onnx/model.onnx'

        # 1. Convert to onnx
        BaseEmbedderInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=onnx_model_path)

        # 2. Inference with onnx session
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "The Eiffel Tower is located in Paris, France.",
            "Python is a popular programming language.",
            "The Great Wall of China is one of the Seven Wonders of the World.",
            "Space exploration has led to many scientific discoveries.",
            "Climate change is a pressing global issue.",
            "The Mona Lisa is a famous painting by Leonardo da Vinci.",
            "Electric cars are becoming more common.",
            "The human brain is an incredibly complex organ."
        ]

        args=AbsInferenceArguments(
            model_name_or_path=model_path,
            onnx_model_path=onnx_model_path,
            trt_model_path=None,
            infer_mode='onnx',
            infer_device=0,
            infer_batch_size=16
        )
        inference_engine_onnx = BaseEmbedderInferenceEngine(args)
        emb_onnx = inference_engine_onnx.inference(sentences, normalize=True, batch_size=5)
        print(emb_onnx.shape)
        print(emb_onnx[0]@ emb_onnx[1].T)
        ```
        
    3. TensorRT

        Use official tool `trtexec` to convert onnx model to TensorRT or use our `BaseEmbedderInferenceEngine.convert_to_tensorrt` which relys on trtexec.

        1. Bash scripts of converting ONNX to TensorRT
            ```bash
            # scripts to convert onnx to tensorrt

            ONNX_PATH='' # onnx model path
            TRT_SAVE_PATH='' # tensorrt model path, dirpath should be created early

            # your tensorrt path here
            TRT_PATH=''

            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_PATH/lib
            export PATH=$PATH:$TRT_PATH/bin


            # Convert ONNX to TensorRT with dynamic shapes. Please refer to official docs for detailed usage
            trtexec --onnx=$ONNX_PATH --saveEngine=$TRT_SAVE_PATH --minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 --optShapes=input_ids:8x128,attention_mask:8x128,token_type_ids:8x128 --maxShapes=input_ids:16x512,attention_mask:16x512,token_type_ids:16x512 --verbose
            ```

        2. Inference with TensorRT
            ```python
            from InfoNexus import AbsInferenceArguments, BaseEmbedderInferenceEngine


            # trt path is path to TensorRT you have downloaded.
            trt_path='/data2/home/angqing/tensorrt/TensorRT-10.7.0.23'

            model_path='/data2/OpenLLMs/bge-base-zh-v1.5'
            trt_model_path ='/data2/OpenLLMs/bge-base-zh-v1.5/trt/model.trt'
            onnx_model_path='/data2/OpenLLMs/bge-base-zh-v1.5/onnx/model.onnx'

            sentences = [
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming the world.",
                "The Eiffel Tower is located in Paris, France.",
                "Python is a popular programming language.",
                "The Great Wall of China is one of the Seven Wonders of the World.",
                "Space exploration has led to many scientific discoveries.",
                "Climate change is a pressing global issue.",
                "The Mona Lisa is a famous painting by Leonardo da Vinci.",
                "Electric cars are becoming more common.",
                "The human brain is an incredibly complex organ."
            ]

            args=AbsInferenceArguments(
                model_name_or_path=model_path,
                onnx_model_path=onnx_model_path,
                trt_model_path=trt_model_path,
                infer_mode='tensorrt',
                infer_device=7,
                infer_batch_size=16
            )

            # Convert onnx to tensorrt. Skip if you already have tensorrt model.  
            BaseEmbedderInferenceEngine.convert_to_tensorrt(args.onnx_model_path, args.trt_model_path, args.infer_batch_size, trt_path=trt_path)

            inference_engine_tensorrt=BaseEmbedderInferenceEngine(args)


            emb_trt=inference_engine_tensorrt.inference(sentences, normalize=True, batch_size=5)
            print(emb_trt.shape)
            print(emb_trt[0]@ emb_trt[1].T)
            ```

2. Reranker

    1. normal
    
        There are two ways of normal inference. The first is to use FlagModel based method, the second is to use inference engine with `infer_mode` setting to `'normal'`

        ```python
        # 1. not use inference engine
        from InfoNexus import TextReranker, AbsInferenceArguments, BaseRerankerInferenceEngine

        # inputs should be Union[Tuple, List[Tuple]]
        qa_pairs = [
            ("What is the capital of France?", "Paris is the capital and most populous city of France."),
            ("Who wrote 'Pride and Prejudice'?","Edison wrote this." ),
            ("What is the largest planet in our solar system?", "May be our mother land."),
            ("Who is the current president of the United States?", "The current president of the United States is Joe Biden."),
            ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
            ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second."),
            ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
            ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world."),
            ("What is the smallest country in the world?", "Vatican City is the smallest country in the world."),
            ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming.")
        ]

        model_name_or_path= '/data2/OpenLLMs/bge-reranker-base'

        model = TextReranker(model_name_or_path=model_name_or_path, normalize=True, use_fp16=True, devices=['cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        score = model.compute_score(qa_pairs)
        print(score)
        # 2. using inference engine
        args=AbsInferenceArguments(
            model_name_or_path=model_name_or_path,
            infer_mode='normal',
            infer_device=0,
            infer_batch_size=16
        )
        args.infer_mode = 'normal'
        inference_engine = BaseRerankerInferenceEngine(args)
        score = inference_engine.inference(qa_pairs, batch_size=10, normalize=True)
        print(score)
        ```

    2. ONNX
        
        Convert pytorch model to onnx first.

        ```python
        from InfoNexus import AbsInferenceArguments, BaseRerankerInferenceEngine
        model_path='/data2/OpenLLMs/bge-reranker-base'
        onnx_model_path='/data2/OpenLLMs/bge-reranker-base/onnx/model.onnx'

        # 1. convert pytorch model to ONNX
        BaseRerankerInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=onnx_model_path)

        # inputs should be Union[Tuple, List[Tuple]]
        qa_pairs = [
            ("What is the capital of France?", "Paris is the capital and most populous city of France."),
            ("Who wrote 'Pride and Prejudice'?","Edison wrote this." ),
            ("What is the largest planet in our solar system?", "May be our mother land."),
            ("Who is the current president of the United States?", "The current president of the United States is Joe Biden."),
            ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
            ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second."),
            ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
            ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world."),
            ("What is the smallest country in the world?", "Vatican City is the smallest country in the world."),
            ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming.")
        ]
        # 2. Inference with onnx session
        args=AbsInferenceArguments(
            model_name_or_path=model_path,
            onnx_model_path=onnx_model_path,
            trt_model_path=None,
            infer_mode='onnx',
            infer_device=0,
            infer_batch_size=16
        )
        inference_engine_onnx = BaseRerankerInferenceEngine(args)
        score = inference_engine_onnx.inference(qa_pairs, normalize=True, batch_size=5)
        print(score)
        ```
        
    3. TensorRT

        Please use official tool `trtexec` to convert onnx model to TensorRT first, or use `BaseRerankerInferenceEngine.convert_to_tensorrt` .

        1. Bash scripts of converting ONNX to TensorRT
            ```bash
            # scripts to convert onnx to tensorrt

            ONNX_PATH='' # onnx model path
            TRT_SAVE_PATH='' # tensorrt model path, dirpath should be created early

            # your tensorrt path here
            TRT_PATH=''

            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_PATH/lib
            export PATH=$PATH:$TRT_PATH/bin


            # Convert ONNX to TensorRT with dynamic shapes. Please refer to official docs for detailed usage
            trtexec --onnx=$ONNX_PATH --saveEngine=$TRT_SAVE_PATH --minShapes=input_ids:1x1,attention_mask:1x1 --optShapes=input_ids:8x128,attention_mask:8x128 --maxShapes=input_ids:16x512,attention_mask:16x512 --verbose
            ```
        
        2. Inference with TensorRT
            ```python
            from InfoNexus import AbsInferenceArguments, BaseRerankerInferenceEngine

            # trt path is path to TensorRT you have downloaded.
            trt_path='/data2/home/angqing/tensorrt/TensorRT-10.7.0.23'

            model_path='/data2/OpenLLMs/bge-reranker-base'
            onnx_model_path='/data2/OpenLLMs/bge-reranker-base/onnx/model.onnx'
            trt_model_path='/data2/OpenLLMs/bge-reranker-base/trt/model.trt'

            qa_pairs = [
                ("What is the capital of France?", "Paris is the capital and most populous city of France."),
                ("Who wrote 'Pride and Prejudice'?","Edison wrote this." ),
                ("What is the largest planet in our solar system?", "May be our mother land."),
                ("Who is the current president of the United States?", "The current president of the United States is Joe Biden."),
                ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
                ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second."),
                ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
                ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world."),
                ("What is the smallest country in the world?", "Vatican City is the smallest country in the world."),
                ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming.")
            ]

            args=AbsInferenceArguments(
                model_name_or_path=model_path,
                onnx_model_path=onnx_model_path,
                trt_model_path=trt_model_path,
                infer_mode='tensorrt',
                infer_device=7,
                infer_batch_size=16
            )

            # Convert onnx model to tensorrt. Skip if you already have tensorrt model.
            BaseRerankerInferenceEngine.convert_to_tensorrt(args.onnx_model_path, args.trt_model_path, args.infer_batch_size, trt_path)

            inference_engine_tensorrt = BaseRerankerInferenceEngine(args)

            score = inference_engine_tensorrt.inference(qa_pairs, normalize=True, batch_size=5)
            print(score)
            ```

## Eval

```bash
BASE_DIR=/data1/home/recstudio/angqing/InfoNexus
EMBEDDER=/data1/home/recstudio/angqing/models/bge-base-zh-v1.5
RERANKER=/data1/home/recstudio/angqing/models/bge-reranker-base
embedder_infer_mode=onnx
reranker_infer_mode=onnx
embedder_onnx_path=$EMBEDDER/onnx/model.onnx
reranker_onnx_path=$RERANKER/onnx/model.onnx
embedder_trt_path=$EMBEDDER/trt/model.trt
reranker_trt_path=$RERANKER/trt/model.trt


cd $BASE_DIR


python -m InfoNexus.evaluation.text_retrieval.airbench \
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
```