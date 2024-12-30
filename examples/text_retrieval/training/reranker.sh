export WANDB_MODE=disabled

train_data='/data2/home/angqing/code/UniRetrieval/examples/text_retrieval/example_data/fiqa.jsonl'

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
    --model_name_or_path /data2/OpenLLMs/bge-reranker-base \
    --cache_dir $HF_HUB_CACHE \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size $train_group_size \
    --query_max_len 256 \
    --passage_max_len 256 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
"

training_args="\
    --output_dir /data2/home/angqing/code/UniRetrieval/checkpoints/test_reranker \
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
    --deepspeed /data2/home/angqing/code/UniRetrieval/examples/text_retrieval/training/ds_stage0.json \
    --logging_steps 1 \
    --save_steps 100 \
"
cd /data2/home/angqing/code/UniRetrieval
cmd="torchrun --nproc_per_node $num_gpus \
    -m UniRetrieval.training.reranker.text_retrieval \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd