"""Shared model, sampling, and data helpers for AudioFlow post-training."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchdiffeq
from torch.utils.data import DataLoader

from audio_flow.lora import inject_lora
from audio_flow.utils import CombinedModel, requires_grad
from train import get_adaptor, get_base, get_dataset, get_sampler


def checkpoint_state(path: str, device: str) -> dict:
    """Read original EMA weights or a resumable post-training snapshot."""

    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        return checkpoint["ema"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def build_model(config: dict, checkpoint: str, posttrained: bool = False):
    """Build the generator and inject LoRA before or after loading weights."""

    device = config["train"]["device"]
    model = CombinedModel(get_base(config), get_adaptor(config)).to(device)
    state = checkpoint_state(checkpoint, device)
    post = config["post_training"]
    requires_grad(model, False)
    lora_options = {
        "rank": int(post["lora_rank"]),
        "alpha": float(post["lora_alpha"]),
        "targets": tuple(post["lora_targets"]),
    }
    if posttrained:
        inject_lora(model.base, **lora_options)
        model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(state, strict=True)
        inject_lora(model.base, **lora_options)
    requires_grad(model.adaptor, True)
    return model


def drop_condition(embeddings: dict, probability: float) -> dict:
    """Drop whole condition embeddings for classifier-free guidance training."""

    if probability <= 0:
        return embeddings
    output = {}
    for key, value in embeddings.items():
        keep = (torch.rand(value.shape[0], device=value.device) >= probability).to(
            value.dtype
        )
        output[key] = value * keep.view(value.shape[0], *([1] * (value.ndim - 1)))
    return output


def real_batch_iterator(config: dict, batch_size: int):
    """Create the infinite-sampler iterator used for real-data replay."""

    dataset = get_dataset(config, split="train", mode="train")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=get_sampler(config, dataset),
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )
    return iter(loader)


def replay_loss(
    model,
    transform,
    iterator,
    matcher,
    condition_dropout: float = 0.0,
):
    """Flow-matching loss on real training data used to prevent reward drift."""

    target, conditions = transform(next(iterator))
    noise = torch.randn_like(target)
    time, state, velocity = matcher.sample_location_and_conditional_flow(noise, target)
    embeddings = drop_condition(model.adaptor(conditions), condition_dropout)
    return F.mse_loss(model.base(time, state, embeddings), velocity)


def save_checkpoint(
    model,
    optimizer,
    path: str | Path,
    step: int,
    method: str,
    config_path: str,
):
    """Atomically save a resumable post-training checkpoint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "meta": {
                "method": method,
                "config": config_path,
                "resumable": True,
            },
        },
        temporary,
    )
    temporary.replace(target)
    print(f"Saved {target}", flush=True)


def sample_latents(model, condition_ids, shape, steps: int, guidance_scale: float = 1.0):
    """Generate matched-noise latents with Euler ODE sampling and CFG."""

    device = next(model.parameters()).device
    condition_ids = torch.as_tensor(condition_ids, dtype=torch.long, device=device)
    noise = torch.randn(len(condition_ids), *shape, device=device)
    embeddings = model.adaptor({"id": condition_ids})
    unconditional = {key: torch.zeros_like(value) for key, value in embeddings.items()}
    times = torch.linspace(0, 1, steps + 1, device=device)

    def vector_field(time, state):
        conditional = model.base(time, state, embeddings)
        if guidance_scale == 1.0:
            return conditional
        empty = model.base(time, state, unconditional)
        return empty + guidance_scale * (conditional - empty)

    with torch.no_grad():
        model.eval()
        trajectory = torchdiffeq.odeint(vector_field, noise, times, method="euler")
    return noise, trajectory[-1]
