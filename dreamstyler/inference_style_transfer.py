# DreamStyler
# Copyright (c) 2024-present NAVER Webtoon
# Apache-2.0

import os
from os.path import join as ospj
import click
import torch
import imageio
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux.processor import Processor
import custom_pipelines



def load_model(sd_path, controlnet_path, embedding_path, placeholder_token="<sks1>", num_stages=6):
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    placeholder_token = [f"{placeholder_token}-T{t}" for t in range(num_stages)]
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError("The tokens are already in the tokenizer")
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    text_encoder.resize_token_embeddings(len(tokenizer))

    learned_embeds = torch.load(embedding_path)
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for token, token_id in zip(placeholder_token, placeholder_token_id):
        token_embeds[token_id] = learned_embeds[token]

    pipeline = custom_pipelines.StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        sd_path,
        controlnet=controlnet,
        text_encoder=text_encoder.to(torch.float16),
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    processor_midas = Processor("depth_midas")

    return pipeline, processor_midas


@click.command()
@click.option("--sd_path")
@click.option("--controlnet_path", default="lllyasviel/control_v11f1p_sd15_depth")
@click.option("--embedding_path")
@click.option("--content_image_path")
@click.option("--saveroot", default="./outputs")
@click.option("--prompt", default="A painting of a city skyline, in the style of {}")
@click.option("--placeholder_token", default="<sks1>")
@click.option("--num_stages", default=6)
@click.option("--num_samples", default=5)
@click.option("--resolution", default=512)
@click.option("--seed")
def style_transfer(
    sd_path=None,
    controlnet_path="lllyasviel/control_v11f1p_sd15_depth",
    embedding_path=None,
    content_image_path=None,
    saveroot="./outputs",
    prompt="A painting of a city skyline, in the style of {}",
    placeholder_token="<sks1>",
    num_stages=6,
    num_samples=5,
    resolution=512,
    seed=None,
):
    os.makedirs(saveroot, exist_ok=True)
    pipeline, processor = load_model(
        sd_path,
        controlnet_path,
        embedding_path,
        placeholder_token,
        num_stages,
    )
    generator = None if seed is None else torch.Generator(device="cuda").manual_seed(seed)
    cross_attention_kwargs = {"num_stages": num_stages}

    content_image = Image.open(content_image_path)
    content_image = content_image.resize((resolution, resolution))
    control_image = processor(content_image, to_pil=True)
    pos_prompt = [prompt.format(f"{placeholder_token}-T{t}") for t in range(num_stages)]

    outputs = []
    # torch.manual_seed(1)    
    for _ in range(num_samples):
        output = pipeline(
            prompt=pos_prompt,
            num_inference_steps=30, 
            generator=generator,
            image=content_image,
            control_image=control_image,
            cross_attention_kwargs=cross_attention_kwargs,
            # strength=0.8,             
            # guidance_scale=7.5      
        ).images[0]
        outputs.append(output)

    outputs = np.concatenate([np.asarray(img) for img in outputs], axis=1)
    save_path = ospj(saveroot, f"{content_image_path.split('/')[-1].split('.')[0]}.png")
    imageio.imsave(save_path, outputs)


if __name__ == "__main__":
    style_transfer()
