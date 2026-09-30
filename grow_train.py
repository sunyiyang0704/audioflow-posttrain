"""GROW: group-relative reward training for the deterministic AudioFlow ODE.

This is advantage-weighted flow matching, not likelihood-ratio GRPO. It keeps
the conditional and unconditional velocity fields close to the SFT reference.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import trange

from audio_flow.lora import trainable_parameters
from audio_flow.rewards import load_reward_function, score_rewards
from audio_flow.utils import parse_yaml, requires_grad
from audio_flow.posttrain import (
    build_model,
    real_batch_iterator,
    replay_loss,
    save_checkpoint,
    sample_latents,
)
from train import get_data_transform


REWARD_AXES = ("clap_condition", "content_enjoyment", "production_quality")


def group_advantages(scores: list[dict], weights: dict, min_std: float, device):
    """Center each learned reward within the candidate group."""

    advantage = torch.zeros(len(scores), device=device)
    for axis in REWARD_AXES:
        values = torch.tensor([row[axis] for row in scores], device=device)
        scale = values.std(unbiased=False).clamp_min(min_std)
        advantage += float(weights[axis]) * (values - values.mean()) / scale
    return advantage - advantage.mean()


def grow_losses(model, reference, ids, latents, advantages, fm):
    """Signed FM objective plus reference anchors for both CFG branches."""

    noise = torch.randn_like(latents)
    time, state, velocity = fm.sample_location_and_conditional_flow(noise, latents)
    embeddings = model.adaptor({"id": ids})
    zeros = {key: torch.zeros_like(value) for key, value in embeddings.items()}
    prediction = model.base(time, state, embeddings)
    per_candidate = (prediction - velocity).square().mean(dim=(1, 2))
    reward_loss = (advantages.detach() * per_candidate).mean()

    with torch.no_grad():
        reference_embeddings = reference.adaptor({"id": ids})
        reference_zeros = {
            key: torch.zeros_like(value) for key, value in reference_embeddings.items()
        }
        reference_conditional = reference.base(time, state, reference_embeddings)
        reference_unconditional = reference.base(time, state, reference_zeros)
    prediction_unconditional = model.base(time, state, zeros)
    anchor_conditional = F.mse_loss(prediction, reference_conditional)
    anchor_unconditional = F.mse_loss(
        prediction_unconditional, reference_unconditional
    )
    anchor = 0.5 * (anchor_conditional + anchor_unconditional)
    return reward_loss, anchor, per_candidate.detach()


def gradient_norm(loss, parameters, retain_graph=True):
    grads = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return sum(
        grad.detach().float().square().sum() for grad in grads if grad is not None
    ).sqrt().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Frozen SFT reference")
    parser.add_argument("--resume-state")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    configs = parse_yaml(args.config)
    post = configs["post_training"]
    device = configs["train"]["device"]
    model = build_model(configs, args.checkpoint, posttrained=True)
    reference = deepcopy(model).eval()
    requires_grad(reference, False)
    if post.get("freeze_adaptor", True):
        requires_grad(model.adaptor, False)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=float(post["grow_lr"]))
    start_step = 1
    if args.resume_state:
        state = torch.load(args.resume_state, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1

    transform = get_data_transform(configs).to(device).eval()
    reward_fn = load_reward_function(post.get("reward", {}), post.get("reward_plugin"))
    iterator = real_batch_iterator(configs, int(post.get("replay_batch_size", 2)))
    fm = ConditionalFlowMatcher(sigma=0.0)
    shape = (configs["base"]["in_dim"], int(post["latent_frames"]))
    group_size = int(post["candidates_per_condition"])
    classes = int(configs["adaptor"]["num_classes"])
    output_dir = Path(post["output_dir"])
    weights = post.get(
        "grow_reward_weights",
        {
            "clap_condition": 0.5,
            "content_enjoyment": 0.2,
            "production_quality": 0.3,
        },
    )
    min_std = float(post.get("grow_min_reward_std", 0.02))

    for step in trange(start_step, int(post["grow_steps"]) + 1, desc="GROW"):
        condition_id = (step - 1) % classes
        ids = torch.full((group_size,), condition_id, dtype=torch.long, device=device)
        _, latents = sample_latents(
            model,
            ids,
            shape,
            int(post["sample_steps"]),
            float(post["guidance_scale"]),
        )
        model.train()
        with torch.inference_mode():
            audios = transform.latent_to_audio(latents).cpu().numpy()
        scores = score_rewards(
            reward_fn, audios, transform.sr, [condition_id] * group_size
        )
        advantages = group_advantages(scores, weights, min_std, device)
        reward_loss, anchor, per_candidate = grow_losses(
            model, reference, ids, latents.detach(), advantages, fm
        )
        supervised = replay_loss(
            model, transform, iterator, fm, float(post["condition_dropout"])
        )
        loss = (
            float(post.get("grow_advantage_weight", 1.0)) * reward_loss
            + float(post.get("grow_anchor_weight", 1.0)) * anchor
            + float(post.get("replay_weight", 0.2)) * supervised
        )
        if step == start_step:
            reward_grad = gradient_norm(reward_loss, parameters)
            replay_grad = gradient_norm(supervised, parameters)
            print(
                f"gradient_check reward={reward_grad:.6f} replay={replay_grad:.6f} "
                f"adv_range=({advantages.min().item():.3f},{advantages.max().item():.3f})",
                flush=True,
            )
            if not (reward_grad > 0 and torch.isfinite(loss)):
                raise RuntimeError("Reward objective has no finite training gradient")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == start_step or step % 20 == 0:
            print(
                f"grow step={step} loss={loss.item():.4f} "
                f"reward={reward_loss.item():.4f} anchor={anchor.item():.4f} "
                f"replay={supervised.item():.4f} score={sum(row['total'] for row in scores)/group_size:.4f} "
                f"fm_range=({per_candidate.min().item():.4f},{per_candidate.max().item():.4f})",
                flush=True,
            )
        if step % int(post.get("grow_checkpoint_every", 100)) == 0:
            save_checkpoint(
                model,
                optimizer,
                output_dir / f"grow_step={step}.pt",
                step,
                "advantage_weighted_flow_matching",
                args.config,
            )


if __name__ == "__main__":
    main()
