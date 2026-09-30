from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import soundfile
import torch
from torch import LongTensor
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
import wandb
from audio_flow.utils import (CombinedModel, LinearWarmUp, parse_yaml,
                              requires_grad, update_ema, logmel)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import tqdm

from train import get_data_transform, get_base, get_adaptor
from audio_flow.datasets.gtzan_vae import GtzanVAE


def sample(args):

    # Arguments
    config_path = args.config
    ckpt_path = args.ckpt_path
    configs = parse_yaml(config_path)
    device = configs["train"].get("device", "cuda")

    data_transform = get_data_transform(configs).to(device)

    # Model
    base = get_base(configs=configs).to(device)
    adaptor = get_adaptor(configs=configs).to(device)
    model = CombinedModel(base, adaptor)

    # Load checkpoint
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        # Accept both the new resumable checkpoint and legacy EMA-only files.
        state_dict = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)

    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Prepare condition
    for id in range(10):

        cond_dict = {
            "id": LongTensor([id]).to(device),
        }

        noise = torch.randn(1, 64, 250).to(device)  # (b, d, t)

        with torch.no_grad():
            model.eval()
            emb_dict = model.adaptor(cond_dict)
            traj = torchdiffeq.odeint(
                lambda t, x: model.base(t, x, emb_dict),
                y0=noise,
                t=torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )

        x_gen = traj[-1]  # (b, d, t)
        gen_audio = data_transform.latent_to_audio(x_gen).data.cpu().numpy()[0]  # (c, l)

        # Visualize logmel
        sr = data_transform.sr
        gen_logmel = logmel(gen_audio, sr)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.matshow(gen_logmel.T, origin='lower', aspect='auto', cmap='jet', vmin=-10, vmax=5)
        out_path = Path(out_dir, "{}.pdf".format(GtzanVAE.IX_TO_LB[id]))
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Write out to {out_path}")

        # Write to audio
        out_path = Path(out_dir, "{}.wav".format(GtzanVAE.IX_TO_LB[id]))
        soundfile.write(file=out_path, data=gen_audio.T, samplerate=sr)
        print(f"Write out to {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path of checkpoint.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="_tmp",
        help="Directory for generated WAV and PDF files.",
    )
    args = parser.parse_args()

    sample(args)
