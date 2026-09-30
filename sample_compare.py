"""Save matched SFT and post-training audio for listening checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from audio_flow.rewards import GTZAN_LABELS
from audio_flow.utils import parse_yaml
from audio_flow.posttrain import build_model, sample_latents
from train import get_data_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="NAME=PATH, repeat for each model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()

    configs = parse_yaml(args.config)
    post = configs["post_training"]
    transform = get_data_transform(configs).to(configs["train"]["device"]).eval()
    shape = (configs["base"]["in_dim"], int(post["latent_frames"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in args.checkpoint:
        name, path = entry.split("=", 1)
        model = build_model(configs, path, posttrained=True).eval()
        for condition_id, label in enumerate(GTZAN_LABELS):
            torch.manual_seed(args.seed + condition_id)
            _, latents = sample_latents(
                model, [condition_id], shape, int(post["sample_steps"]),
                float(post["guidance_scale"]),
            )
            with torch.inference_mode():
                audio = transform.latent_to_audio(latents)[0].cpu().numpy()
            sf.write(output_dir / f"{condition_id:02d}_{label}_{name}.wav",
                     audio.T, transform.sr)
        print(f"Saved matched audio for {name} to {output_dir}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
