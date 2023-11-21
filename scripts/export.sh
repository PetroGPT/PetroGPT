model=/mnt/sdb/models/Baichuan2-7B-Chat/
template=baichuan2
lora_dir=output/baichuan2_sft
outdir=output/export/baichuan2_sft

mkdir -p $outdir
petrogpt-export  --model_name_or_path $model \
    --template $template \
    --finetuning_type lora \
    --checkpoint_dir $lora_dir \
    --export_dir $outdir