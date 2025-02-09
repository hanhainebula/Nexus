export WANDB_MODE=disabled

base_dir='/data1/home/recstudio/haoran/Nexus'
train_data='/data1/home/recstudio/haoran/angqing_temp/data/fiqa.jsonl'
model_name_or_path='/data1/home/recstudio/angqing/models/bge-base-zh-v1.5'
ckpt_save_dir='/data1/home/recstudio/haoran/angqing_temp/ckpt/test_embedder'

deepspeed='/data1/home/recstudio/haoran/Nexus/examples/text_retrieval/training/ds_stage0.json'
# set large epochs and small batch size for testing
num_train_epochs=1
per_device_train_batch_size=8

# set num_gpus to 2 for testing
num_gpus=1

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path $model_name_or_path \
    --cache_dir $HF_HUB_CACHE \
"

data_args="\
    --train_data $train_data \
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
    --output_dir $ckpt_save_dir \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed $deepspeed \
    --logging_steps 1 \
    --save_steps 100 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
"


cd $base_dir

cmd="torchrun --nproc_per_node $num_gpus \
    -m Nexus.training.embedder.text_retrieval \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
