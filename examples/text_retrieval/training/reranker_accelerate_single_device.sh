export WANDB_MODE=disabled

BASE_DIR='/root/Nexus'

MODEL_NAME_OR_PATH='/data1/home/recstudio/angqing/models/bge-reranker-base'
TRAIN_DATA='/data1/home/recstudio/haoran/angqing_temp/data/fiqa.jsonl'
CKPT_SAVE_DIR='/data1/home/recstudio/haoran/angqing_temp/ckpt/test_reranker_accelerate'
ACCELERATE_CONFIG='/root/Nexus/examples/text_retrieval/training/single_node_single_device.json'


# set large epochs and small batch size for testing
num_train_epochs=1
per_device_train_batch_size=8
gradient_accumulation_steps=1
train_group_size=8

# set num_gpus to 2 for testing
num_gpus=1

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
    --logging_steps 1 \
    --save_steps 100 \
"
cd $BASE_DIR

cmd="accelerate launch --config_file $ACCELERATE_CONFIG \
    Nexus/training/reranker/text_retrieval/__main__.py \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd