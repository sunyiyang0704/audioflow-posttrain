"""LoRA supervised fine-tuning with condition dropout for text-to-music."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import trange

from audio_flow.lora import trainable_parameters
from audio_flow.posttrain import (
    build_model,
    drop_condition,
    real_batch_iterator,
    save_checkpoint,
)
from audio_flow.utils import parse_yaml
from train import get_data_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Original EMA checkpoint")
    parser.add_argument("--resume-state", help="A previous sft_step=*.pt snapshot")
    parser.add_argument("--steps", type=int, help="Override the configured total steps")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = parse_yaml(args.config)
    post = config["post_training"]
    device = config["train"]["device"]
    source = args.resume_state or args.checkpoint
    model = build_model(config, source, posttrained=bool(args.resume_state))
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=float(post["sft_lr"]))
    start = 1
    if args.resume_state:
        state = torch.load(args.resume_state, map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"]) + 1

    transform = get_data_transform(config).to(device)
    iterator = real_batch_iterator(config, int(post["batch_size"]))
    matcher = ConditionalFlowMatcher(sigma=0.0)
    total_steps = int(args.steps or post["sft_steps"])
    checkpoint_every = int(post.get("sft_checkpoint_every", 1000))
    output_dir = Path(post["output_dir"])
    model.train()

    for step in trange(start, total_steps + 1, desc="LoRA SFT"):
        target, conditions = transform(next(iterator))
        noise = torch.randn_like(target)
        time, state, velocity = matcher.sample_location_and_conditional_flow(noise, target)
        embeddings = drop_condition(
            model.adaptor(conditions), float(post["condition_dropout"])
        )
        loss = F.mse_loss(model.base(time, state, embeddings), velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == start or step % int(post.get("log_every", 20)) == 0:
            print(f"sft step={step} loss={loss.item():.4f}", flush=True)
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            save_checkpoint(
                model,
                optimizer,
                output_dir / f"sft_step={step}.pt",
                step,
                "lora_sft",
                args.config,
            )

    if total_steps < start or checkpoint_every <= 0 or total_steps % checkpoint_every:
        save_checkpoint(
            model,
            optimizer,
            output_dir / f"sft_step={total_steps}.pt",
            total_steps,
            "lora_sft",
            args.config,
        )


if __name__ == "__main__":
    main()
