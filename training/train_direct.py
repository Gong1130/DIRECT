#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import logging
import math
import os
import shutil
from pathlib import Path
import copy
import itertools
import re

import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import torchvision.utils as vutils
import torch.nn as nn

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxFillPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from transformers import AutoModel, AutoProcessor

from dataset.basedataset import direct_collate_fn
from dataset.composed_dataset import ComposedDataset

import trellis.models as trellis_models

from safetensors.torch import load_file, save_file

from direct.geometry import render_gaussian_from_slat_arbitrary_size, paste_geometry_condition

from direct import DirectPipeline
from direct.layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
from direct.transformer_flux import FluxTransformer2DModelwithcond


if is_wandb_available():
    import wandb
    
logger = get_logger(__name__, log_level="INFO")

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)(?:-epoch-(\d+))?$")


def checkpoint_sort_key(path):
    name = Path(path).name
    match = CHECKPOINT_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid checkpoint directory name: {name}")
    step = int(match.group(1))
    epoch = int(match.group(2) or 0)
    return step, epoch, name


def list_checkpoint_dirs(output_dir):
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    checkpoints = [
        path
        for path in output_path.iterdir()
        if path.is_dir() and CHECKPOINT_RE.match(path.name)
    ]
    return sorted(checkpoints, key=checkpoint_sort_key)


def prune_checkpoints(output_dir, total_limit, keep_slots=1):
    if total_limit is None:
        return
    if total_limit < 1:
        raise ValueError("--checkpoints_total_limit must be at least 1")

    checkpoints = list_checkpoint_dirs(output_dir)
    max_existing = max(total_limit - keep_slots, 0)
    removing_checkpoints = checkpoints[: max(len(checkpoints) - max_existing, 0)]
    if removing_checkpoints:
        logger.info(
            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        )
        logger.info("removing checkpoints: " + ", ".join(path.name for path in removing_checkpoints))

    for checkpoint in removing_checkpoints:
        shutil.rmtree(checkpoint)


def resolve_resume_checkpoint(output_dir, resume_from_checkpoint):
    if resume_from_checkpoint is None:
        return None
    if resume_from_checkpoint == "latest":
        checkpoints = list_checkpoint_dirs(output_dir)
        return checkpoints[-1] if checkpoints else None

    checkpoint_path = Path(resume_from_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(output_dir) / checkpoint_path.name
    return checkpoint_path


def dtype_to_config_name(dtype):
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    return "float32"


def save_direct_config(output_dir, args, lora_ranks, lora_alphas, text_lora_config, weight_dtype):
    config = {
        "model_type": "direct_pipeline",
        "flux_model": args.base_model_path,
        "siglip_model": args.siglip_model_path,
        "torch_dtype": dtype_to_config_name(weight_dtype),
        "lora": {
            "ranks": lora_ranks,
            "alphas": lora_alphas,
            "weights": [1 for _ in range(args.num_loras)],
            "n_loras": args.num_loras,
            "double_blocks": 19,
            "single_blocks": 38,
            "text": text_lora_config,
        },
        "condition_embedder": {
            "input_dim": 64,
        },
        "pooled_image_projector": {
            "input_dim": 1152,
            "output_dim": 768,
        },
        "image_projector": {
            "input_dim": 1152,
            "output_dim": 4096,
        },
        "weight_files": {
            "lora": "lora.safetensors",
            "condition_embedder": "condition_embedder.safetensors",
            "x_embedder": "x_embedder.safetensors",
            "time_text_embed": "time_text_embed.safetensors",
            "pooled_image_projector": "pooled_image_projector.safetensors",
            "image_projector": "image_projector.safetensors",
        },
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


@torch.no_grad()
def log_training_visualization(direct_pipeline, gaussian_decoder, args, accelerator, global_step, batch, weight_dtype):
    logger.info("Running train visualization...")

    N = min(batch["target_image"].shape[0], 4)
    target_image = (batch["target_image"][:N].permute(0, 3, 1, 2).to(weight_dtype) + 1) / 2
    masked_target_image = (batch["masked_target_image"][:N].permute(0, 3, 1, 2).to(weight_dtype) + 1) / 2
    inpainting_mask = batch["inpainting_mask"][:N].permute(0, 3, 1, 2).to(weight_dtype)
    object_mask = batch["object_mask"][:N].permute(0, 3, 1, 2).to(weight_dtype)
    ref_image = (batch["masked_ref_image"][:N].permute(0, 3, 1, 2).to(weight_dtype) + 1) / 2
    size = tuple(target_image.shape[2:4])
    target_slat, target_w2c = batch["target_slat"][:N].to(torch.float32), batch["target_w2c"][:N].to(torch.float32)
    target_gaussian, target_gaussian_mask = render_gaussian_from_slat_arbitrary_size(target_slat, target_w2c, size, gaussian_decoder, return_mask=True)
    target_gaussian = torch.clamp(target_gaussian, 0, 1).to(weight_dtype)
    target_gaussian_mask = target_gaussian_mask.to(weight_dtype)
    full_masked_target_image = batch["full_masked_target_image"][:N].permute(0, 3, 1, 2)

    pasted_image, updated_inpainting_mask = paste_geometry_condition(
            masked_target_image,    # [0, 1]
            object_mask,            # binary
            target_gaussian,        # [0, 1]
            target_gaussian_mask,   # binary
            inpainting_mask         # binary
        )
    
    output = direct_pipeline(
        composite_image=pasted_image,
        inpaint_mask=updated_inpainting_mask,
        reference_image=ref_image,
        geometry_image=target_gaussian,
        context_image=full_masked_target_image,
        num_samples=1,
        num_inference_steps=28,
        output_type='pt',
        seed=args.seed,
        guidance_scale=30,
        height=size[0],
        width=size[1],
        use_autocast=True
    )
    
    ref_image_norm = torch.clamp(ref_image, 0, 1)
    pasted_image_norm = torch.clamp(pasted_image, 0, 1)
    target_image_norm = torch.clamp(target_image, 0, 1)
    target_gaussian_norm = target_gaussian
    output = torch.clamp(output, 0, 1)
    full_masked_norm = full_masked_target_image.to(weight_dtype) / 255.0
    full_masked_norm = torch.nn.functional.interpolate(
        full_masked_norm, 
        size=size, 
        mode="bilinear", 
        align_corners=False
    )
    full_masked_norm = torch.clamp(full_masked_norm, 0, 1)

    combined = torch.cat([ref_image_norm, target_gaussian_norm, pasted_image_norm, target_image_norm, output, full_masked_norm], dim=3)
    combined = combined.detach().cpu()
    grid = vutils.make_grid(combined, nrow=1, padding=2)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_image("train_visualization", grid, global_step)
        elif tracker.name == "wandb":
            tracker.log({"train_visualization": wandb.Image(grid, caption=f"step {global_step}")})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    free_memory()
    return grid

def parse_args():
    parser = argparse.ArgumentParser(description="Train DIRECT for geometry-conditioned image inpainting.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--siglip_model_path",
        type=str,
        default="google/siglip2-so400m-patch14-384",
        help="Path or Hugging Face repo id for the SigLIP image encoder.",
    )
    parser.add_argument(
        "--trellis_gaussian_decoder_path",
        type=str,
        default="microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16",
        help="Path or Hugging Face repo id for the TRELLIS Gaussian decoder.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        default=None,
        required=True,
        help="Path to the YAML file containing the global and individual dataset configurations.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=None,
        help=("Save a checkpoint of the training state every X epochs."),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--train_visualization_steps",
        type=int,
        default=0,
        help="Run training-batch visualization every X steps. Set to 0 to disable.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="direct",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--visualize_train_start",
        action="store_true",
        help="If set, run training-batch visualization at the start of training."
    )
    # lora parameters
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument(
        "--pretrained_checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="Path to a DIRECT checkpoint used for initialization.",
    )
    parser.add_argument("--num_loras", type=int, default=2, help="Number of LoRA branches.")
    # text lora
    parser.add_argument(
        "--text_lora_rank",
        type=int,
        default=128,
        help="Rank for Text LoRA matrices."
    )
    parser.add_argument(
        "--text_lora_alpha",
        type=float,
        default=128,
        help="Alpha for Text LoRA scaling."
    )
    # flow matching parameters
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    # guidance scale
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=30.0,
        help="Guidance scale used by the FLUX transformer during training.",
    )
    # training modules
    parser.add_argument(
        "--ref_cfg_drop_ratio",
        type=float,
        default=0.0
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKL.from_pretrained(
        args.base_model_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModelwithcond.from_pretrained(
        args.base_model_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    gaussian_decoder = trellis_models.from_pretrained(args.trellis_gaussian_decoder_path)
    image_encoder = AutoModel.from_pretrained(args.siglip_model_path).vision_model.eval()
    siglip_processor = AutoProcessor.from_pretrained(args.siglip_model_path, use_fast=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    gaussian_decoder.to(accelerator.device)


    pooled_image_projector = nn.Linear(1152, 768)
    image_projector = nn.Linear(1152, 4096)
    condition_embedder = nn.Linear(64, transformer.inner_dim)
    with torch.no_grad():
        condition_embedder.weight.copy_(
            transformer.x_embedder.weight[:, :64].to(
                device=condition_embedder.weight.device,
                dtype=condition_embedder.weight.dtype,
            )
        )
        condition_embedder.bias.copy_(
            transformer.x_embedder.bias.to(
                device=condition_embedder.bias.device,
                dtype=condition_embedder.bias.dtype,
            )
        )
        
    def ensure_list_param(param, count):
        if not isinstance(param, list):
            param = [param]
        if len(param) == 1 and count > 1:
            print(f"Broadcasting parameter {param} to length {count}...")
            return param * count
        if len(param) != count:
            raise ValueError(f"The length of the provided parameter ({len(param)}) does not match num_loras ({count})!")
        return param

    lora_ranks = ensure_list_param(args.lora_rank, args.num_loras)
    lora_alphas = ensure_list_param(args.lora_alpha, args.num_loras)

    text_lora_config = {
        "rank": args.text_lora_rank,
        "alpha": args.text_lora_alpha,
        "token_length": 729
    }
    lora_attn_procs = {}
    lora_device = transformer.device
    double_blocks_idx = list(range(19))
    single_blocks_idx = list(range(38))
    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        layer_index = int(match.group(1)) if match else -1
        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                dim=3072, ranks=lora_ranks, network_alphas=lora_alphas, lora_weights=[1 for _ in range(args.num_loras)], device=lora_device, dtype=weight_dtype,
                n_loras=args.num_loras, text_lora_config=text_lora_config
            )
        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
            lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                dim=3072, ranks=lora_ranks, network_alphas=lora_alphas, lora_weights=[1 for _ in range(args.num_loras)], device=lora_device, dtype=weight_dtype,
                n_loras=args.num_loras, text_lora_config=text_lora_config
            )
        else:
            lora_attn_procs[name] = attn_processor
    transformer.set_attn_processor(lora_attn_procs)

    if args.pretrained_checkpoint_path is not None:
        condition_embedder.load_state_dict(load_file(os.path.join(args.pretrained_checkpoint_path, "condition_embedder.safetensors")))
        pooled_image_projector.load_state_dict(load_file(os.path.join(args.pretrained_checkpoint_path, "pooled_image_projector.safetensors")), strict=True)
        image_projector.load_state_dict(load_file(os.path.join(args.pretrained_checkpoint_path, "image_projector.safetensors")), strict=True)
        transformer.load_state_dict(load_file(os.path.join(args.pretrained_checkpoint_path, "x_embedder.safetensors")), strict=False)
        transformer.load_state_dict(load_file(os.path.join(args.pretrained_checkpoint_path, "time_text_embed.safetensors")), strict=False)
        lora_state_dict = load_file(os.path.join(args.pretrained_checkpoint_path, "lora.safetensors"))
        missing, unexpected = transformer.load_state_dict(lora_state_dict, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded pretrained checkpoint from: {args.pretrained_checkpoint_path}")
            if len(missing) > 0 or len(unexpected) > 0:
                print(f"  Warning - Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            else:
                print("  All keys matched successfully.")

    # Freeze base modules, then open only the DIRECT training surface.
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    gaussian_decoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.train()
    for n, param in transformer.named_parameters():
        if "_lora" in n:
            param.requires_grad = True
    transformer.time_text_embed.requires_grad_(True)
    transformer.x_embedder.requires_grad_(True)
    condition_embedder.train()
    pooled_image_projector.train()
    image_projector.train()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if accelerator.is_main_process:
        print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'M parameters')

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unwrapped_transformer = unwrap_model(transformer)
            transformer_state = unwrapped_transformer.state_dict()
            lora_state_dict = {k: v for k, v in transformer_state.items() if "_lora" in k}
            save_file(lora_state_dict, os.path.join(output_dir, "lora.safetensors"))
            save_file(unwrap_model(condition_embedder).state_dict(), os.path.join(output_dir, "condition_embedder.safetensors"))
            save_file(unwrap_model(pooled_image_projector).state_dict(), os.path.join(output_dir, "pooled_image_projector.safetensors"))
            save_file(unwrap_model(image_projector).state_dict(), os.path.join(output_dir, "image_projector.safetensors"))

            x_embedder_state = {k: v for k, v in transformer_state.items() if "x_embedder" in k}
            save_file(x_embedder_state, os.path.join(output_dir, "x_embedder.safetensors"))
            time_text_state = {k: v for k, v in transformer_state.items() if "time_text_embed" in k}
            save_file(time_text_state, os.path.join(output_dir, "time_text_embed.safetensors"))
            save_direct_config(output_dir, args, lora_ranks, lora_alphas, text_lora_config, weight_dtype)
        weights.clear()

    def load_model_hook(models, input_dir):
        unwrap_model(condition_embedder).load_state_dict(
            load_file(os.path.join(input_dir, "condition_embedder.safetensors"))
        )
        unwrap_model(pooled_image_projector).load_state_dict(
            load_file(os.path.join(input_dir, "pooled_image_projector.safetensors")),
            strict=True,
        )
        unwrap_model(image_projector).load_state_dict(
            load_file(os.path.join(input_dir, "image_projector.safetensors")),
            strict=True,
        )
        unwrapped_transformer = unwrap_model(transformer)
        unwrapped_transformer.load_state_dict(load_file(os.path.join(input_dir, "x_embedder.safetensors")), strict=False)
        unwrapped_transformer.load_state_dict(load_file(os.path.join(input_dir, "time_text_embed.safetensors")), strict=False)
        lora_path = os.path.join(input_dir, "lora.safetensors")
        if os.path.exists(lora_path):
            unwrapped_transformer.load_state_dict(load_file(lora_path), strict=False)
        models.clear()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes / 256.0
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    condition_embedder_parameters = list(filter(lambda p: p.requires_grad, condition_embedder.parameters()))
    pooled_image_projector_parameters = list(filter(lambda p: p.requires_grad, pooled_image_projector.parameters()))
    image_projector_parameters = list(filter(lambda p: p.requires_grad, image_projector.parameters()))

    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    condition_embedder_parameters_with_lr = {"params": condition_embedder_parameters, "lr": args.learning_rate}
    pooled_image_projector_parameters_with_lr = {"params": pooled_image_projector_parameters, "lr": args.learning_rate}
    image_projector_parameters_with_lr = {"params": image_projector_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr, condition_embedder_parameters_with_lr, pooled_image_projector_parameters_with_lr, image_projector_parameters_with_lr]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    train_dataset = ComposedDataset(
        config_path=args.dataset_config_path,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=direct_collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels

    # Scheduler and math around the number of steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    transformer, condition_embedder, pooled_image_projector, image_projector, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, condition_embedder, pooled_image_projector, image_projector, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_step = None
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        checkpoint_path = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
        if checkpoint_path is None or not checkpoint_path.exists():
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {checkpoint_path.name}")
            accelerator.load_state(str(checkpoint_path))
            global_step = checkpoint_sort_key(checkpoint_path)[0]

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def build_train_visualization_pipeline():
        direct_pipeline = DirectPipeline.from_training_components(
            flux_model_path=args.base_model_path,
            transformer=unwrap_model(transformer),
            vae=vae,
            condition_embedder=unwrap_model(condition_embedder),
            image_encoder=image_encoder,
            siglip_processor=siglip_processor,
            pooled_image_projector=unwrap_model(pooled_image_projector),
            image_projector=unwrap_model(image_projector),
            device=accelerator.device,
            torch_dtype=weight_dtype,
            cache_dir=args.cache_dir,
            revision=args.revision,
            variant=args.variant,
        )
        direct_pipeline.pipe.set_progress_bar_config(disable=True)
        return direct_pipeline
    
    has_guidance = unwrap_model(transformer).config.guidance_embeds
    did_initial_visualization = False
    for epoch in range(first_epoch, args.num_train_epochs):
        active_dataloader = train_dataloader
        if resume_step is not None and epoch == first_epoch and resume_step > 0:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)

        for step, batch in enumerate(active_dataloader):
            models_to_accumulate = [transformer, condition_embedder]
            if accelerator.is_main_process:
                if global_step == 0 and args.visualize_train_start and not did_initial_visualization:
                    did_initial_visualization = True
                    direct_pipeline = build_train_visualization_pipeline()
                    output = log_training_visualization(
                        direct_pipeline,
                        gaussian_decoder,
                        args,
                        accelerator,
                        global_step,
                        batch,
                        weight_dtype,
                    )
                    save_folder = os.path.join(args.output_dir, "image_log")
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, f"{global_step:06d}.png")
                    vutils.save_image(output, save_path)
                    del direct_pipeline
                    free_memory()
            with accelerator.accumulate(models_to_accumulate):
                # Convert images to latent space
                with torch.no_grad():
                    # gt
                    target_image = batch["target_image"].permute(0, 3, 1, 2).to(weight_dtype)
                    target_latent = vae.encode(target_image).latent_dist.sample()
                    target_latent = (target_latent - vae_config_shift_factor) * vae_config_scaling_factor
                    bsz = target_latent.shape[0]
                    size = tuple(target_image.shape[2:4])

                    target_w2c = batch["target_w2c"].to(torch.float32)
                    target_slat = batch["target_slat"].to(torch.float32)
                    target_gaussian_image, target_gaussian_mask = render_gaussian_from_slat_arbitrary_size(target_slat, target_w2c, size, decoder=gaussian_decoder, return_mask=True)
                    target_gaussian_image = target_gaussian_image.to(weight_dtype)
                    target_gaussian_mask = target_gaussian_mask.to(weight_dtype)

                    masked_target_image = (batch["masked_target_image"].permute(0, 3, 1, 2).to(weight_dtype) + 1) / 2

                    object_mask = batch["object_mask"].permute(0, 3, 1, 2).to(weight_dtype)
                    inpainting_mask = batch["inpainting_mask"].permute(0, 3, 1, 2).to(weight_dtype)

                    pasted_image, inpainting_mask = paste_geometry_condition(
                        masked_target_image,
                        object_mask,
                        target_gaussian_image,
                        target_gaussian_mask,
                        inpainting_mask
                    )
                    masked_target_latent = vae.encode(pasted_image * 2 - 1).latent_dist.sample()
                    masked_target_latent = (masked_target_latent - vae_config_shift_factor) * vae_config_scaling_factor
                    
                    masked_ref_image = batch["masked_ref_image"].permute(0, 3, 1, 2).to(weight_dtype)
                    if args.ref_cfg_drop_ratio > 0.0:
                        random_probs = torch.rand(bsz, device=masked_ref_image.device)
                        drop_mask = random_probs < args.ref_cfg_drop_ratio
                        masked_ref_image[drop_mask] = 0.0
                    ref_latent = vae.encode(masked_ref_image).latent_dist.sample()
                    ref_latent = (ref_latent - vae_config_shift_factor) * vae_config_scaling_factor

                    target_gaussian_latent = vae.encode(target_gaussian_image * 2 - 1).latent_dist.sample()
                    target_gaussian_latent = (target_gaussian_latent - vae_config_shift_factor) * vae_config_scaling_factor

                    siglip_input = batch["full_masked_target_image"].permute(0, 3, 1, 2)
                    siglip_input = siglip_processor(images=siglip_input, return_tensors="pt").to(masked_ref_image.device)
                    siglip_output = image_encoder(**siglip_input)
                
                pooled_prompt_embeds = pooled_image_projector(siglip_output.pooler_output)
                prompt_embeds = image_projector(siglip_output.last_hidden_state)
                text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=masked_ref_image.device, dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                inpainting_mask = inpainting_mask[:, 0, :, :]
                inpainting_mask = inpainting_mask.view(
                    bsz, target_latent.shape[2], vae_scale_factor, target_latent.shape[3], vae_scale_factor
                )  # batch_size, height, 8, width, 8
                inpainting_mask = inpainting_mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
                inpainting_mask = inpainting_mask.reshape(
                    bsz, vae_scale_factor * vae_scale_factor, target_latent.shape[2], target_latent.shape[3]
                ) 
                inpainting_mask = FluxFillPipeline._pack_latents(
                    inpainting_mask,
                    bsz,
                    vae_scale_factor * vae_scale_factor,
                    inpainting_mask.shape[2],
                    inpainting_mask.shape[3]
                )
                
                all_image_ids = []
                for i, latent in enumerate([target_latent, ref_latent, target_gaussian_latent]):
                    ids = FluxFillPipeline._prepare_latent_image_ids(
                        batch_size=latent.shape[0],
                        height=latent.shape[2] // 2,
                        width=latent.shape[3] // 2,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    ids[..., 0] = i
                    all_image_ids.append(ids)
                latent_image_ids = torch.cat(all_image_ids, dim=0)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(target_latent)

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=target_latent.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=target_latent.ndim, dtype=target_latent.dtype)
                noisy_model_input = (1.0 - sigmas) * target_latent + sigmas * noise
                packed_latent_chunks = []
                for lat in [noisy_model_input, masked_target_latent, ref_latent, target_gaussian_latent]:
                    packed_latent_chunks.append(
                        FluxFillPipeline._pack_latents(
                            lat,
                            batch_size=lat.shape[0],
                            num_channels_latents=lat.shape[1],
                            height=lat.shape[2],
                            width=lat.shape[3],
                        )
                    )
                packed_noisy_model_input = torch.cat([packed_latent_chunks[0], packed_latent_chunks[1], inpainting_mask], dim=2)
                
                cond_model_input = torch.cat([*packed_latent_chunks[2:]], dim=1)
                cond_hidden_states = condition_embedder(cond_model_input)

                guidance = None
                if has_guidance:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(noisy_model_input.shape[0])

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    cond_hidden_states=cond_hidden_states,
                    # FluxTransformer2DModel expects timesteps scaled to [0, 1].
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxFillPipeline._unpack_latents(
                    model_pred,
                    height=target_latent.shape[2] * vae_scale_factor,
                    width=target_latent.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - target_latent

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(
                            transformer.parameters(),
                            condition_embedder.parameters(),
                            pooled_image_projector.parameters(),
                            image_projector.parameters(),
                        )
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        prune_checkpoints(args.output_dir, args.checkpoints_total_limit, keep_slots=1)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    if args.train_visualization_steps > 0 and global_step % args.train_visualization_steps == 0:
                        direct_pipeline = build_train_visualization_pipeline()
                        output = log_training_visualization(
                            direct_pipeline,
                            gaussian_decoder,
                            args,
                            accelerator,
                            global_step,
                            batch,
                            weight_dtype,
                        )
                        save_folder = os.path.join(args.output_dir, "image_log")
                        os.makedirs(save_folder, exist_ok=True)
                        save_path = os.path.join(save_folder, f"{global_step:06d}.png")
                        vutils.save_image(output, save_path)
                        del direct_pipeline
                        free_memory()
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break

        if args.checkpointing_epochs is not None and (epoch + 1) % args.checkpointing_epochs == 0:
            if accelerator.is_main_process:
                prune_checkpoints(args.output_dir, args.checkpoints_total_limit, keep_slots=1)
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-epoch-{epoch + 1}")
                os.makedirs(save_path, exist_ok=True)
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

    if accelerator.is_main_process and global_step > 0:
        prune_checkpoints(args.output_dir, args.checkpoints_total_limit, keep_slots=1)
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        os.makedirs(save_path, exist_ok=True)
        accelerator.save_state(save_path)
        logger.info(f"Saved final state to {save_path}")

    accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
