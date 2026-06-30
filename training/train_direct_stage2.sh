#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

accelerate launch --config_file training/acc_config_8gpu.yaml \
  --mixed_precision="bf16" training/train_direct.py \
  --base_model_path='black-forest-labs/FLUX.1-Fill-dev' \
  --siglip_model_path='google/siglip2-so400m-patch14-384' \
  --trellis_gaussian_decoder_path='microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16' \
  --dataset_config_path='dataset_config/direct_stage2_1024.yaml' \
  --train_batch_size=1 \
  --dataloader_num_workers=0 \
  --max_train_steps=40000 \
  --train_visualization_steps=4000 \
  --checkpointing_steps=4000 \
  --learning_rate=1e-4 \
  --output_dir="outputs/direct-stage2" \
  --seed 231 \
  --tracker_project_name direct \
  --lora_rank 128 \
  --lora_alpha 128 \
  --num_loras 2 \
  --guidance_scale 30 \
  --visualize_train_start \
  --ref_cfg_drop_ratio 0.1 \
  --text_lora_rank 128 \
  --text_lora_alpha 128 \
  --gradient_checkpointing \
  --pretrained_checkpoint_path='outputs/direct-stage1/checkpoint-200000'
