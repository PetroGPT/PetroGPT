# PetroGPT
石油领域大语言模型



install：

```shell
git clone https://github.com/PetroGPT/PetroGPT
cd PetroGPT
conda create -n petrogpt python=3.10
conda activate petrogpt
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple 
pip install -e .
```



pt:

```shell
#bash scripts/train_pt.sh

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
```

sft:

```shell
#bash scripts/train_sft.sh

export CUDA_VISIBLE_DEVICES=0
model=/mnt/sdb/models/Baichuan2-7B-Chat/
template=baichuan2
lora_weight=W_pack
data=petro_sft
outdir=output/baichuan2_sft
mkdir -p $outdir

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

```



export:

```shell
#bash scripts/export.sh

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
```

eval:

```shell
#bash scripts/eval.sh

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

```





Our work is primarily based on the foundation of numerous open-source contributions. Thanks to the following open source projects

- [LLaMa-Efficient-Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)