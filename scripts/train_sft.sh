export CUDA_VISIBLE_DEVICES=0

model=/mnt/sdb/models/Baichuan2-7B-Chat/
template=baichuan2
lora_weight=W_pack
data=petro_sft
outdir=output/baichuan2_sft
mkdir -p $outdir

# wandb： pip install wandb
# export WANDB_API_KEY=your_key
# export WANDB_ENTITY=your_entity
# export WANDB_PROJECT=your_project


# lora fine-tune
petrogpt-train --stage sft \
    --model_name_or_path $model \
    --do_train \
    --dataset $data \
    --template $template \
    --finetuning_type lora \
    --lora_target $lora_weight \
    --output_dir $outdir \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 > $outdir/train.log 2>&1

# full params fine-tune
#petrogpt-train --stage sft \
#    --model_name_or_path $model \
#    --do_train \
#    --dataset $data \
#    --template $template \
#    --finetuning_type full \
#    --output_dir $outdir \
#    --overwrite_cache \
#    --per_device_train_batch_size 4 \
#    --gradient_accumulation_steps 4 \
#    --lr_scheduler_type cosine \
#    --logging_steps 10 \
#    --save_steps 1000 \
#    --learning_rate 5e-5 \
#    --num_train_epochs 3.0 \
#    --plot_loss \
#    --bf16 > $outdir/train.log 2>&1


# deepspeed lora fine-tune
#deepspeed --num_gpus 2 --master_port=9901 petrogpt-train \
#    --deepspeed petrogpt/config/ds_config2.json \
#    --stage sft --model_name_or_path $model \
#    --do_train \
#    --dataset $data \
#    --template $template \
#    --finetuning_type full \
#    --lora_target $lora_weight \
#    --output_dir $outdir \
#    --overwrite_cache \
#    --per_device_train_batch_size 4 \
#    --gradient_accumulation_steps 4 \
#    --lr_scheduler_type cosine \
#    --logging_steps 10 \
#    --save_steps 1000 \
#    --learning_rate 5e-5 \
#    --num_train_epochs 3.0 \
#    --plot_loss \
#    --bf16 > $outdir/train.log 2>&1


# deepspeed full fine-tune
#deepspeed --num_gpus 2 --master_port=9901 petrogpt-train \
#    --deepspeed petrogpt/config/ds_config3.json \
#    --stage sft --model_name_or_path $model \
#    --do_train \
#    --dataset $data \
#    --template $template \
#    --finetuning_type full \
#    --output_dir $outdir \
#    --overwrite_cache \
#    --per_device_train_batch_size 4 \
#    --gradient_accumulation_steps 4 \
#    --lr_scheduler_type cosine \
#    --logging_steps 10 \
#    --save_steps 1000 \
#    --learning_rate 5e-5 \
#    --num_train_epochs 3.0 \
#    --plot_loss \
#    --bf16 > $outdir/train.log 2>&1