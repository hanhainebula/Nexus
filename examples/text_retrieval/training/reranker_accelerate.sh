export WANDB_MODE=disabled

BASE_DIR='/data1/home/recstudio/angqing/UniRetrieval'

MODEL_NAME_OR_PATH='/data2/OpenLLMs/bge-reranker-base'
TRAIN_DATA="/data1/home/recstudio/angqing/UniRetrieval/eval_scripts/training/text_retrieval/example_data/fiqa.jsonl"
CKPT_SAVE_DIR='/data2/home/angqing/code/UniRetrieval/checkpoints/test_reranker'
DEEPSPEED_DIR='/data1/home/recstudio/angqing/UniRetrieval/eval_scripts/training/ds_stage0.json'
ACCELERATE_CONFIG='/data1/home/recstudio/angqing/UniRetrieval/eval_scripts/training/text_retrieval/accelerate_config_multi.json'


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
    UniRetrieval/training/reranker/text_retrieval/__main__.py \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd