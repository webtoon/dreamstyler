#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import os
import argparse
import logging
from os.path import join as ospj

import diffusers
import accelerate
import numpy as np
import transformers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision

from PIL import Image
from tqdm import trange
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

if is_wandb_available():
    import wandb

check_min_version("0.20.0")
logger = get_logger(__name__)


class DreamStylerDataset(torch.utils.data.Dataset):
    template = "A painting in the style of {}"

    def __init__(
        self,
        image_path,
        tokenizer,
        size=512,
        repeats=100,
        prob_flip=0.5,
        placeholder_tokens="*",
        center_crop=False,
        is_train=True,
        num_stages=1,
        context_prompt=None,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_tokens = placeholder_tokens
        self.center_crop = center_crop
        self.prob_flip = prob_flip
        self.repeats = repeats if is_train else 1
        self.num_stages = num_stages

        if not isinstance(self.placeholder_tokens, list):
            self.placeholder_tokens = [self.placeholder_token]

        self.flip = torchvision.transforms.RandomHorizontalFlip(p=self.prob_flip)

        self.image_path = image_path
        self.prompt = self.template if context_prompt is None else context_prompt

    def __getitem__(self, index):
        image = Image.open(self.image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)
        prompt = self.prompt

        tokens = []
        for t in range(self.num_stages):
            placeholder_string = self.placeholder_tokens[t]
            prompt_t = prompt.format(placeholder_string)

            tokens.append(
                self.tokenizer(
                    prompt_t,
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
            )

        if self.center_crop:
            h, w = image.shape[0], image.shape[1]
            min_hw = min(h, w)
            image = image[
                (h - min_hw) // 2 : (h + min_hw) // 2,
                (w - min_hw) // 2 : (w + min_hw) // 2,
            ]

        image = Image.fromarray(image)
        image = image.resize((self.size, self.size), resample=Image.LANCZOS)
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        return {
            "input_ids": tokens,
            "pixel_values": image,
        }

    def __len__(self):
        return self.repeats


def train(opt):
    accelerator = init_accelerator_and_logger(logger, opt)
    (
        train_dataset,
        train_dataloader,
        placeholder_tokens,
        placeholder_token_ids,
        tokenizer,
        text_encoder,
        noise_scheduler,
        optimizer,
        lr_scheduler,
        vae,
        unet,
        weight_dtype,
    ) = init_model_and_dataset(accelerator, logger, opt)

    # do we need this?
    if opt.resume_from_checkpoint:
        raise NotImplementedError

    # keep original embeddings as reference
    orig_embeds_params = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight.data.clone()
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Total optimization steps = {opt.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {opt.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")

    text_encoder.train()
    for step in trange(opt.max_train_steps, disable=not accelerator.is_local_main_process):
        try:
            batch = next(iters)
        except (UnboundLocalError, StopIteration, TypeError):
            iters = iter(train_dataloader)
            batch = next(iters)

        with accelerator.accumulate(text_encoder):
            # convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype))
            latents = latents.latent_dist.sample().detach() * vae.config.scaling_factor

            # sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Dreamstyler: get index in stage (T) axis
            max_timesteps = noise_scheduler.config.num_train_timesteps
            index_stage = (timesteps / max_timesteps * opt.num_stages).long()

            # add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # get the text embedding for conditioning
            # Dreamstyler: batch["input_ids"] is [T x bsz x 77]-dim shape
            # and if bsz > 1, timesteps have multiple arbitary t values
            # so that input_ids variable should be proprocessed
            # to be matched to appropriate timesteps
            input_ids = torch.empty_like(batch["input_ids"][0])
            for n in range(bsz):
                input_ids[n] = batch["input_ids"][index_stage[n]][n]
            encoder_hidden_states = text_encoder(input_ids)[0].to(weight_dtype)

            # predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # let's make sure we don't update any embedding weights
            # besides the newly added token
            index_no_updates = ~torch.isin(
                torch.arange(len(tokenizer)),
                torch.tensor(placeholder_token_ids),
            )
            with torch.no_grad():
                emb1 = accelerator.unwrap_model(text_encoder).get_input_embeddings()
                emb2 = orig_embeds_params[index_no_updates]
                emb1.weight[index_no_updates] = emb2

        if accelerator.sync_gradients:
            if accelerator.is_main_process and (step + 1) % opt.save_steps == 0:
                save(
                    accelerator,
                    text_encoder,
                    placeholder_tokens,
                    placeholder_token_ids,
                    step + 1,
                    opt,
                )

    accelerator.wait_for_everyone()
    save(
        accelerator,
        text_encoder,
        placeholder_tokens,
        placeholder_token_ids,
        "final",
        opt,
    )
    accelerator.end_training()


def init_accelerator_and_logger(logger, opt):
    logging_dir = ospj(opt.output_dir, opt.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=opt.output_dir,
        logging_dir=logging_dir,
    )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        mixed_precision=opt.mixed_precision,
        log_with=opt.report_to,
        project_config=accelerator_project_config,
    )

    if opt.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it"
                " for logging during training."
            )

    # make one log on every process with the configuration for debugging.
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

    # handle the repository creation
    if accelerator.is_main_process:
        if opt.output_dir is not None:
            os.makedirs(opt.output_dir, exist_ok=True)

    os.makedirs(ospj(opt.output_dir, "embedding"), exist_ok=True)
    return accelerator


def init_model_and_dataset(accelerator, logger, opt, without_dataset=False):
    if opt.seed is not None:
        set_seed(opt.seed)

    if opt.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(opt.tokenizer_name)
    elif opt.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            opt.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )

    noise_scheduler = DDPMScheduler.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=opt.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="vae",
        revision=opt.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        opt.pretrained_model_name_or_path,
        subfolder="unet",
        revision=opt.revision,
    )

    # DreamStyler: TODO: support multi-vector TI
    if opt.num_vectors > 1:
        raise NotImplementedError

    # DreamStyler: add new textual embeddings
    placeholder_tokens = [
        f"{opt.placeholder_token}-T{t}" for t in range(opt.num_stages)
    ]
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {opt.placeholder_token}."
            " Please pass a different `placeholder_token` that is not already in the tokenizer."
        )

    # convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(opt.initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # initialize the newly added placeholder token
    # with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # freeze vae and unet and text encoder (except for the token embeddings)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if opt.gradient_checkpointing:
        # keep unet in train mode if we are using gradient checkpointing to save memory.
        # the dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if opt.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import version
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs."
                    " If you observe problems during training, please update xFormers"
                    " to at least 0.0.17."
                    " See https://huggingface.co/docs/diffusers/main/en/optimization/xformers"
                    " for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if opt.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if opt.scale_lr:
        opt.learning_rate = (
            opt.learning_rate
            * opt.gradient_accumulation_steps
            * opt.train_batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=opt.learning_rate,
        betas=(opt.adam_beta1, opt.adam_beta2),
        weight_decay=opt.adam_weight_decay,
        eps=opt.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        opt.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=opt.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=opt.max_train_steps * accelerator.num_processes,
        num_cycles=opt.lr_num_cycles,
    )

    if without_dataset:
        train_dataset, train_dataloader = None, None
        text_encoder, optimizer, lr_scheduler = accelerator.prepare(
            text_encoder,
            optimizer,
            lr_scheduler,
        )
    else:
        train_dataset = DreamStylerDataset(
            image_path=opt.train_image_path,
            tokenizer=tokenizer,
            size=opt.resolution,
            placeholder_tokens=placeholder_tokens,
            repeats=opt.max_train_steps,
            center_crop=opt.center_crop,
            is_train=True,
            context_prompt=opt.context_prompt,
            num_stages=opt.num_stages,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.train_batch_size,
            shuffle=True,
            num_workers=opt.dataloader_num_workers,
        )
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

    text_encoder, optimizer, lr_scheduler = accelerator.prepare(
        text_encoder,
        optimizer,
        lr_scheduler,
    )

    # for mixed precision training we cast all non-trainable weigths
    # (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference,
    # keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # we need to initialize the trackers we use, and also store our configuration.
    # the trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreamstyler", config=vars(opt))

    return (
        train_dataset,
        train_dataloader,
        placeholder_tokens,
        placeholder_token_ids,
        tokenizer,
        text_encoder,
        noise_scheduler,
        optimizer,
        lr_scheduler,
        vae,
        unet,
        weight_dtype,
    )


def save(
    accelerator,
    text_encoder,
    placeholder_tokens,
    placeholder_token_ids,
    prefix,
    opt,
):
    prefix = f"{prefix:04d}" if isinstance(prefix, int) else prefix

    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings()
    embeds_dict = {}
    for token, token_id in zip(placeholder_tokens, placeholder_token_ids):
        embeds_dict[token] = learned_embeds.weight[token_id].detach().cpu()
    torch.save(embeds_dict, ospj(opt.output_dir, "embedding", f"{prefix}.bin"))


def get_options():
    parser = argparse.ArgumentParser()

    # DreamStyler arguments
    parser.add_argument(
        "--context_prompt",
        type=str,
        default=None,
        help="Additional context prompt to enhance training performance.",
    )
    parser.add_argument(
        "--num_stages",
        type=int,
        default=6,
        help="The number of the stages (denoted as T) used in multi-stage TI.",
    )

    # original textual inversion arguments
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_image_path",
        type=str,
        default=None,
        required=True,
        help="A path of training image.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token",
        type=str,
        default="painting",
        help="A token to use as initializer word.",
    )
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="style",
        help="Choose between 'object' and 'style'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dreamstyler",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the"
            " train/validation dataset will be resized to this resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
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
        help=(
            "Whether or not to use gradient checkpointing"
            " to save memory at the expense of slower backward pass."
        ),
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
        help=(
            "Scale the learning rate by the number of GPUs,"
            " gradient accumulation steps, and batch size."
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            "The scheduler type to use. Choose between"
            " ['linear', 'cosine', 'cosine_with_restarts', 'polynomial',"
            " 'constant', 'constant_with_warmup']"
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=(
            "Number of subprocesses to use for data loading."
            " 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
            " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs."
            " Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            "The integration to report the results and logs to."
            ' Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`.'
            ' Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=5,
        help=(
            "Number of images that should be generated"
            " during validation with `validation_prompt`.",
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates."
            " These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
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
            "Whether training should be resumed from a previous checkpoint."
            " Use a path saved by `--checkpointing_steps`,"
            ' or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help=(
            "If specified save the checkpoint not in `safetensors` format,"
            " but in original PyTorch format instead.",
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank not in (-1, args.local_rank):
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":
    opt = get_options()
    train(opt)
