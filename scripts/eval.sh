export CUDA_VISIBLE_DEVICES=0
model=/mnt/sdb/models/Baichuan2-7B-Chat/
template=baichuan2
data=petro_sft
lora_dir=output/baichuan2_sft
outdir=output/baichuan2_sft_predict

mkdir -p $outdir
petrogpt-eval --stage sft \
    --model_name_or_path $model   \
    --do_predict \
    --dataset $data \
    --finetuning_type lora \
    --output_dir  $outdir \
    --per_device_eval_batch_size 4 \
    --predict_with_generate --checkpoint_dir  $lora_dir \
    --template $template --max_new_tokens  2048
