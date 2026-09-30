"""Experimental SDE Flow-GRPO with a separately reported waveform verifier.

The policy is the Gaussian transition of a stochastic version of AudioFlow's
noise-to-data flow. This is not the older MSE-as-log-probability surrogate.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import trange

from audio_flow.lora import trainable_parameters
from audio_flow.rewards import load_reward_function, score_rewards, signal_rewards
from audio_flow.utils import parse_yaml, requires_grad
from audio_flow.posttrain import (
    build_model,
    real_batch_iterator,
    replay_loss,
    save_checkpoint,
)
from train import get_data_transform


def transition_parameters(state, velocity, time: float, dt: float, noise_scale: float):
    """Euler-Maruyama transition preserving the noise-to-data flow marginals.

    For x_t=(1-t) noise+t data, score=-(x_t-t*v_t)/(1-t). Adding
    sigma^2/2 * score to the ODE drift and sigma*dW preserves its marginals.
    The midpoint avoids the singular endpoint in sigma^2=a^2(1-t)/t.
    """
    if not 0.0 < time < 1.0 or dt <= 0 or noise_scale <= 0:
        raise ValueError("Expected 0 < time < 1, dt > 0, noise_scale > 0")
    a2 = noise_scale**2
    drift = (1.0 + 0.5 * a2) * velocity - (0.5 * a2 / time) * state
    variance = a2 * (1.0 - time) / time * dt
    return state + dt * drift, variance


def gaussian_log_ratio(action, new_mean, old_mean, variance):
    """Exact log probability ratio of the full Gaussian transition action."""
    delta = ((action - old_mean).square() - (action - new_mean).square())
    return (delta / (2.0 * variance)).flatten(1).sum(1)


def group_advantages(rewards: torch.Tensor, min_std: float = 0.02):
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(min_std)


def guided_velocity(model, state, time: float, embeddings, guidance: float):
    t = torch.full((state.shape[0],), time, device=state.device)
    conditional = model.base(t, state, embeddings)
    if guidance == 1.0:
        return conditional
    empty = {key: torch.zeros_like(value) for key, value in embeddings.items()}
    unconditional = model.base(t, state, empty)
    return unconditional + guidance * (conditional - unconditional)


@torch.no_grad()
def rollout(model, ids, shape, steps: int, guidance: float, noise_scale: float):
    model.eval()
    embeddings = model.adaptor({"id": ids})
    state = torch.randn(len(ids), *shape, device=ids.device)
    dt = 1.0 / steps
    trajectory = []
    for index in range(steps):
        time = (index + 0.5) * dt
        velocity = guided_velocity(model, state, time, embeddings, guidance)
        mean, variance = transition_parameters(state, velocity, time, dt, noise_scale)
        next_state = mean + math.sqrt(variance) * torch.randn_like(state)
        trajectory.append((state, next_state, mean, time, variance))
        state = next_state
    return state, trajectory


def score_group(
    reward_fn,
    audios,
    sample_rate,
    condition_id,
    verified_weight,
    reward_mode="hybrid",
    verifier_floor=0.78,
    verifier_penalty=0.4,
):
    learned_rows = score_rewards(
        reward_fn, audios, sample_rate, [condition_id] * len(audios)
    )
    rows = []
    for audio, scores in zip(audios, learned_rows):
        # Exact waveform measurements are verifiable; genre/aesthetics remain
        # learned proxy scores and are reported separately.
        verified = signal_rewards(audio, sample_rate, condition_id)["total"]
        learned = (
            0.5 * scores["clap_condition"]
            + 0.2 * scores["content_enjoyment"]
            + 0.3 * scores["production_quality"]
        )
        if reward_mode == "hybrid":
            objective = (1.0 - verified_weight) * learned + verified_weight * verified
        elif reward_mode == "total_with_verifier_floor":
            objective = scores["total"] - verifier_penalty * max(
                0.0, verifier_floor - verified
            )
        else:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")
        rows.append(
            {
                **scores,
                "verified_signal": verified,
                "learned_reward": learned,
                "hybrid_reward": objective,
            }
        )
    return rows


def policy_loss(
    model,
    reference,
    ids,
    trajectory,
    selected,
    advantages,
    guidance,
    clip,
    kl_weight,
):
    embeddings = model.adaptor({"id": ids})
    ref_embeddings = reference.adaptor({"id": ids})
    surrogates, kls, log_ratios = [], [], []
    for index in selected:
        state, action, old_mean, time, variance = trajectory[index]
        dt = 1.0 / len(trajectory)
        new_velocity = guided_velocity(model, state, time, embeddings, guidance)
        noise_scale = math.sqrt(variance * time / ((1.0 - time) * dt))
        new_mean, _ = transition_parameters(
            state, new_velocity, time, dt, noise_scale
        )
        with torch.no_grad():
            ref_velocity = guided_velocity(reference, state, time, ref_embeddings, guidance)
            ref_mean, _ = transition_parameters(
                state, ref_velocity, time, dt, noise_scale
            )
        log_ratio = gaussian_log_ratio(action, new_mean, old_mean, variance)
        ratio = log_ratio.clamp(-20.0, 20.0).exp()
        surrogate = torch.minimum(
            ratio * advantages,
            ratio.clamp(1.0 - clip, 1.0 + clip) * advantages,
        )
        # Normalize KL by latent elements; the policy ratio above remains the
        # exact full-action ratio.
        kl = ((new_mean - ref_mean).square() / (2.0 * variance)).flatten(1).mean(1)
        surrogates.append(surrogate.mean())
        kls.append(kl.mean())
        log_ratios.append(log_ratio.detach().abs().mean())
    pg = -torch.stack(surrogates).mean()
    kl = torch.stack(kls).mean()
    return (
        pg + kl_weight * kl,
        pg.detach(),
        kl.detach(),
        torch.stack(log_ratios).mean(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume-state")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    config = parse_yaml(args.config)
    post = config["post_training"]
    rl = post["flow_grpo"]
    device = config["train"]["device"]
    model = build_model(config, args.checkpoint, posttrained=True)
    reference = deepcopy(model).eval()
    requires_grad(reference, False)
    requires_grad(model.adaptor, False)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=float(rl["lr"]))
    start = 1
    if args.resume_state:
        state = torch.load(args.resume_state, map_location=device)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"]) + 1
    transform = get_data_transform(config).to(device).eval()
    reward_fn = load_reward_function(post.get("reward", {}), post.get("reward_plugin"))
    iterator = real_batch_iterator(config, int(post["replay_batch_size"]))
    matcher = ConditionalFlowMatcher(sigma=0.0)
    out = Path(rl["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    trace = out / "train_rewards.jsonl"
    shape = (int(config["base"]["in_dim"]), int(post["latent_frames"]))
    count = int(rl["group_size"])
    n_steps = int(args.steps or rl["steps"])
    for step in trange(start, n_steps + 1, desc="SDE Flow-GRPO"):
        condition_id = (step - 1) % int(config["adaptor"]["num_classes"])
        ids = torch.full((count,), condition_id, dtype=torch.long, device=device)
        latents, trajectory = rollout(
            model,
            ids,
            shape,
            int(rl["sample_steps"]),
            float(post["guidance_scale"]),
            float(rl["noise_scale"]),
        )
        with torch.inference_mode():
            audios = transform.latent_to_audio(latents).cpu().numpy()
        rows = score_group(
            reward_fn,
            audios,
            transform.sr,
            condition_id,
            float(rl["verified_weight"]),
            reward_mode=rl.get("reward_mode", "hybrid"),
            verifier_floor=float(rl.get("verifier_floor", 0.78)),
            verifier_penalty=float(rl.get("verifier_penalty", 0.4)),
        )
        rewards = torch.tensor([row["hybrid_reward"] for row in rows], device=device)
        advantages = group_advantages(rewards).detach()
        selected = torch.randperm(len(trajectory))[
            : int(rl["train_timesteps"])
        ].tolist()
        for _ in range(int(rl["ppo_epochs"])):
            model.eval()
            objective, pg, kl, ratio_abs = policy_loss(
                model,
                reference,
                ids,
                trajectory,
                selected,
                advantages,
                float(post["guidance_scale"]),
                float(rl["clip"]),
                float(rl["kl_weight"]),
            )
            replay = replay_loss(
                model, transform, iterator, matcher, float(post["condition_dropout"])
            )
            loss = (
                float(rl["policy_weight"]) * objective
                + float(rl["replay_weight"]) * replay
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(rl["max_grad_norm"])
            )
            optimizer.step()
        record = {
            "step": step,
            "condition_id": condition_id,
            "hybrid_mean": rewards.mean().item(),
            "verified_mean": sum(row["verified_signal"] for row in rows) / count,
            "learned_mean": sum(row["learned_reward"] for row in rows) / count,
            "adv_std": advantages.std(unbiased=False).item(),
            "pg": pg.item(),
            "kl_per_element": kl.item(),
            "abs_log_ratio": ratio_abs.item(),
            "replay": replay.item(),
            "grad_norm": float(grad_norm),
            "candidate_scores": rows,
        }
        with trace.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if step == start or step % 10 == 0:
            print(json.dumps(record), flush=True)
        if step % int(rl["checkpoint_every"]) == 0:
            save_checkpoint(
                model,
                optimizer,
                out / f"flow_grpo_step={step}.pt",
                step,
                "sde_flow_grpo_hybrid_rlvr",
                args.config,
            )


if __name__ == "__main__":
    main()
