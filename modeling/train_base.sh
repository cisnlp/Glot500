WANDB_DISABLED=true python -m torch.distributed.launch --nproc_per_node=8 run.py \
  --model_name_or_path xlm-roberta-base \
  --train_file /FILE/FOR/TRAIN \
  --tokenizer_name /FILE/FOR/TOKENIZER \
  --output_dir /DIR/FOR/OUTPUT \
  --cache_dir /DIR/FOR/CACHE \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 4 \
  --fp16 True \
  --do_train \
  --num_train_epochs 100 \
  --save_steps 10000 \
  --ddp_timeout 259200 \

