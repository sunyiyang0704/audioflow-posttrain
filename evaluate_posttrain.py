"""Paired, fixed-noise evaluation for independent post-training snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audio_flow.rewards import load_reward_function, score_rewards
from audio_flow.utils import parse_yaml
from audio_flow.posttrain import build_model, sample_latents
from train import get_data_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="NAME=PATH, repeat for every snapshot")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--candidates", type=int, default=4)
    args = parser.parse_args()

    configs = parse_yaml(args.config)
    post = configs["post_training"]
    transform = get_data_transform(configs).to(configs["train"]["device"])
    reward_fn = load_reward_function(post.get("reward", {}), post.get("reward_plugin"))
    shape = (configs["base"]["in_dim"], int(post["latent_frames"]))
    result = {"seed": args.seed, "candidates": args.candidates, "models": {}}

    for item in args.checkpoint:
        name, path = item.split("=", 1)
        print(f"Evaluating {name}: {path}", flush=True)
        model = build_model(configs, path, posttrained=True).eval()
        rows = []
        for condition_id in range(configs["adaptor"]["num_classes"]):
            torch.manual_seed(args.seed + condition_id)
            ids = [condition_id] * args.candidates
            _, latents = sample_latents(
                model, ids, shape, int(post["sample_steps"]),
                float(post["guidance_scale"]),
            )
            with torch.inference_mode():
                audios = transform.latent_to_audio(latents).cpu().numpy()
            scores = score_rewards(reward_fn, audios, transform.sr, ids)
            rows.extend(
                {"condition_id": condition_id, "candidate": index, **score}
                for index, score in enumerate(scores)
            )
        keys = ["total", "clap_condition", "content_enjoyment",
                "production_quality", "signal_quality"]
        means = {key: float(np.mean([row[key] for row in rows])) for key in keys}
        result["models"][name] = {"checkpoint": path, "means": means, "rows": rows}
        print(f"{name}: {means}", flush=True)
        del model
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
