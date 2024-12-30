# Examples for text_retrieval
## Train
1. Embedder
    ```bash
    torchrun --nproc_per_node 2 \
        -m FlagEmbedding.finetune.embedder.encoder_only.base \
        --model_name_or_path BAAI/bge-large-en-v1.5 \
        --cache_dir ./cache/model \
        --train_data ./example_data/retrieval \
                    ./example_data/sts/sts.jsonl \
                    ./example_data/classification-no_in_batch_neg \
                    ./example_data/clustering-no_in_batch_neg \
        --cache_path ./cache/data \
        --train_group_size 8 \
        --query_max_len 512 \
        --passage_max_len 512 \
        --pad_to_multiple_of 8 \
        --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
        --query_instruction_format '{}{}' \
        --knowledge_distillation False \
        --output_dir ./test_encoder_only_base_bge-large-en-v1.5 \
        --overwrite_output_dir \
        --learning_rate 1e-5 \
        --fp16 \
        --num_train_epochs 2 \
        --per_device_train_batch_size 2 \
        --dataloader_drop_last True \
        --warmup_ratio 0.1 \
        --gradient_checkpointing \
        --deepspeed ../ds_stage0.json \
        --logging_steps 1 \
        --save_steps 1000 \
        --negatives_cross_device \
        --temperature 0.02 \
        --sentence_pooling_method cls \
        --normalize_embeddings True \
        --kd_loss_type kl_div
    ```

2. Reranker
    ```bash
    torchrun --nproc_per_node 2 \
        -m FlagEmbedding.finetune.reranker.encoder_only.base \
        --model_name_or_path BAAI/bge-reranker-v2-m3 \
        --cache_dir ./cache/model \
        --train_data ./example_data/normal/examples.jsonl \
        --cache_path ./cache/data \
        --train_group_size 8 \
        --query_max_len 512 \
        --passage_max_len 512 \
        --pad_to_multiple_of 8 \
        --knowledge_distillation False \
        --output_dir ./test_encoder_only_base_bge-reranker-base \
        --overwrite_output_dir \
        --learning_rate 6e-5 \
        --fp16 \
        --num_train_epochs 2 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --dataloader_drop_last True \
        --warmup_ratio 0.1 \
        --gradient_checkpointing \
        --weight_decay 0.01 \
        --deepspeed ../ds_stage0.json \
        --logging_steps 1 \
        --save_steps 1000
    ```

Detailed scripts are in ./training

## Inference
1. Embedder
    1. normal
    
        There are two ways of normal inference. The first is to use FlagModel based method, the second is to use inference engine with `infer_mode` setting to `'normal'`

        ```python
        # 1. not use inference engine
        # TODO pesudo import, should change
        from UniRetrieval import FlagModel, AbsInferenceArguments, BaseEmbedderInferenceEngine

        sentences_1='The quick brown fox jumps over the lazy dog.'
        sentences_2='Artificial intelligence is transforming the world.'

        model = FlagModel(model_name_or_path='/data2/OpenLLMs/bge-base-zh-v1.5', use_fp16=True, devices=['cuda:1','cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        embeddings_1 = model.encode(sentences_1)
        embeddings_2 = model.encode(sentences_2)
        similarity = embeddings_1 @ embeddings_2.T
        print(similarity)

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
        embeddings_1=inference_engine.inference(sentences_1, batch_size=10, normalize=True)
        ```
    2. ONNX
        
        Convert pytorch model to onnx first.

        ```python
        # 1. convert pytorch model to ONNX
        from UniRetrieval import AbsInferenceArguments, BaseEmbedderInferenceEngine
        model_path='path to pytorch model'
        onnx_model_path='path to save onnx model'
        BaseEmbedderInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=args.onnx_model_path)
        
        # 2. Inference with onnx session
        sentences='The quick brown fox jumps over the lazy dog.'
        args=AbsInferenceArguments(
            model_name_or_path=model_path,
            onnx_model_path=onnx_model_path,
            trt_model_path=None,
            infer_mode='onnx',
            infer_device=0,
            infer_batch_size=16
        )
        inference_engine_onnx = BaseEmbedderInferenceEngine(args)
        embeddings = inference_engine_onnx.inference(sentences, normalize=True)
        ```
        
    3. TensorRT

        Please use official tool `trtexec` to convert onnx model to TensorRT first. 

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
            from UniRetrieval import AbsInferenceArguments, BaseEmbedderInferenceEngine
            model_path = 'path to model'
            trt_model_path ='path to trt model'
            
            sentences='The quick brown fox jumps over the lazy dog.'

            args=AbsInferenceArguments(
                model_name_or_path=model_path,
                onnx_model_path=onnx_model_path,
                trt_model_path=trt_model_path,
                infer_mode='tensorrt',
                infer_device=0,
                infer_batch_size=16
            )
            inference_engine_tensorrt=BaseEmbedderInferenceEngine(args)
            embeddings=inference_engine_tensorrt.inference(sentences, normalize=True)
            ```

2. Reranker

    1. normal
    
        There are two ways of normal inference. The first is to use FlagModel based method, the second is to use inference engine with `infer_mode` setting to `'normal'`

        ```python
        # 1. not use inference engine
        # TODO pesudo import, should change
        from UniRetrieval import FlagReranker, AbsInferenceArguments, BaseRerankerInferenceEngine

        # inputs should be Union[Tuple, List[Tuple]]
        qa_pair = ("What is the capital of France?", "Paris is the capital and most populous city of France.")

        model_name_or_path= ''

        model = FlagReranker(model_name_or_path=model_name_or_path, use_fp16=True, devices=['cuda:1','cuda:0']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        score = model.compute_score(qa_pair)

        # 2. using inference engine
        args=AbsInferenceArguments(
            model_name_or_path=model_name_or_path,
            infer_mode='normal',
            infer_device=0,
            infer_batch_size=16
        )
        args.infer_mode = 'normal'
        inference_engine = BaseRerankerInferenceEngine(args)
        score = inference_engine.inference(qa_pair, batch_size=10, normalize=True)
        ```
    2. ONNX
        
        Convert pytorch model to onnx first.

        ```python
        # 1. convert pytorch model to ONNX
        from UniRetrieval import AbsInferenceArguments, BaseRerankerInferenceEngine
        model_path='path to pytorch model'
        onnx_model_path='path to save onnx model'
        BaseRerankerInferenceEngine.convert_to_onnx(model_name_or_path=model_path, onnx_model_path=args.onnx_model_path)

        # inputs should be Union[Tuple, List[Tuple]]
        qa_pair = ("What is the capital of France?", "Paris is the capital and most populous city of France.")        

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
        score = inference_engine_onnx.inference(qa_pair, normalize=True)
        ```
        
    3. TensorRT

        Please use official tool `trtexec` to convert onnx model to TensorRT first. 

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
            from UniRetrieval import AbsInferenceArguments, BaseRerankerInferenceEngine
            model_path = 'path to model'
            trt_model_path ='path to trt model'
            
            qa_pair = ("What is the capital of France?", "Paris is the capital and most populous city of France.")   

            args=AbsInferenceArguments(
                model_name_or_path=model_path,
                onnx_model_path=onnx_model_path,
                trt_model_path=trt_model_path,
                infer_mode='tensorrt',
                infer_device=0,
                infer_batch_size=16
            )
            inference_engine_tensorrt = BaseRerankerInferenceEngine(args)
            score = inference_engine_tensorrt.inference(qa_pair, normalize=True)
            ```

Normal
ONNX
Tensorrt
## Eval
Custom
