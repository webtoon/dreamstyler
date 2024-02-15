# DreamStyler
# Copyright (c) 2024-present NAVER Webtoon
# Apache-2.0

import os
from os.path import join as ospj
import click
import torch
import numpy as np
import imageio
from diffusers import DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import custom_pipelines


def load_model(sd_path, embedding_path, placeholder_token="<sks1>", num_stages=6):
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")

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

    pipeline = custom_pipelines.StableDiffusionPipeline.from_pretrained(
        sd_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
        weight_dtype=torch.float32,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    return pipeline


@click.command()
@click.option("--sd_path")
@click.option("--embedding_path")
@click.option("--saveroot", default="./outputs")
@click.option("--prompt", default="A painting of a dog in the style of {}")
@click.option("--placeholder_token", default="<sks1>")
@click.option("--num_stages", default=6)
@click.option("--use_sc_guidance", is_flag=True)
@click.option("--sty_gamma", default=0.5)
@click.option("--con_gamma", default=3.0)
@click.option("--neg_gamma", default=5.0)
@click.option("--num_samples", default=5)
@click.option("--seed")
def t2i(
    sd_path=None,
    embedding_path=None,
    saveroot="./outputs",
    prompt="A painting of a dog in the style of {}",
    placeholder_token="<sks1>",
    num_stages=6,
    use_sc_guidance=False,
    sty_gamma=0.5,
    con_gamma=3.0,
    neg_gamma=5.0,
    num_samples=5,
    seed=None,
):
    os.makedirs(saveroot, exist_ok=True)

    pipeline = load_model(sd_path, embedding_path, placeholder_token, num_stages)
    generator = None if seed is None else torch.Generator(device="cuda").manual_seed(seed)
    cross_attention_kwargs = {
        "num_stages": num_stages,
        "use_sc_guidance": use_sc_guidance,
        "sty_gamma": sty_gamma,
        "con_gamma": con_gamma,
        "neg_gamma": neg_gamma,
    }

    pos_prompt = [prompt.format(f"{placeholder_token}-T{t}") for t in range(num_stages)]
    neg_prompt = (
        "low resolution, poorly drawn, worst quality, low quality,"
        " normal quality, blurry image, artifact"
    )

    if use_sc_guidance:
        prompt_null_sty = pos_prompt.replace("in the style of {}", "")
        prompt_null_con = "A painting in the style of {}"
        prompt_null_con = [prompt_null_con.format(f"{placeholder_token}-T{t}") for t in range(num_stages)]
        
        cross_attention_kwargs["prompt_null_style"] = prompt_null_sty
        cross_attention_kwargs["prompt_null_context"] = prompt_null_con

    outputs = []
    for _ in range(num_samples):
        outputs.append(pipeline(
            prompt=pos_prompt,
            num_inference_steps=25,
            generator=generator,
            negative_prompt=neg_prompt,
            cross_attention_kwargs=cross_attention_kwargs,
        ).images[0])

    outputs = np.concatenate([np.asarray(img) for img in outputs], axis=1)
    save_path = ospj(saveroot, f"{prompt.replace('in the style of {}', '')}.png")
    imageio.imsave(save_path, outputs)


if __name__ == "__main__":
    t2i()
