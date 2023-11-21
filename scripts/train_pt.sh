export CUDA_VISIBLE_DEVICES=0

model=/mnt/sdb/models/Baichuan2-7B-Base/
lora_weight=W_pack
data=petro_pt
outdir=output/baichuan2_pt
mkdir -p $outdir

# lora pretrain
petrogpt-train --stage pt \
    --model_name_or_path $model \
    --do_train \
    --dataset $data \
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


# full params pretrain
#petrogpt-train --stage pt \
#    --model_name_or_path $model \
#    --do_train \
#    --dataset $data \
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




# deepspeed lora pretrain
#deepspeed --num_gpus 2 --master_port=9901 petrogpt-train \
#    --deepspeed petrogpt/config/ds_config2.json \
#    --stage pt \
#    --model_name_or_path $model \
#    --do_train \
#    --dataset $data \
#    --finetuning_type lora \
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


# deepspeed full params pretrain
#deepspeed --num_gpus 2 --master_port=9901 petrogpt-train \
#    --deepspeed petrogpt/config/ds_config2.json \
#    --stage pt \
#    --model_name_or_path $model \
#    --do_train \
#    --dataset $data \
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

