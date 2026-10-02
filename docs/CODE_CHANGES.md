# Implementation notes

## Training fixes

- Training now performs exactly `training_steps` optimizer updates and avoids
  step-0 validation/checkpointing.
- `latest_train.pt` stores model, EMA, optimizer, scheduler, and step for
  resumable training; older EMA-only checkpoints remain loadable.
- The configured FP32/BF16 precision is respected.
- LoRA parameters inherit the wrapped layer's device and dtype.
- Sampling accepts an output directory and both old and new checkpoint formats.
- The VAE repository is configurable through `AUDIOFLOW_VAE_REPO_ID`.

## Post-training

- **SFT:** rank-16 LoRA on attention and FFN layers, 0.1 condition dropout,
  and CFG 2.0 at inference.
- **Advantage-Weighted Matching (AWM):** four candidates per genre,
  group-normalized CLAP/Audiobox rewards, frozen-reference anchoring, and
  GTZAN replay.
- **Flow-GRPO:** stochastic flow trajectories with Gaussian log-ratios, PPO
  clipping, reference KL, waveform verification, and GTZAN replay.

All stages save resumable checkpoints. Evaluation uses matched initial noise for
paired comparisons.

## Results

- SFT improved the fixed-seed score from 0.2577 to 0.4290 (+66.5%).
- AWM 800 improved the paired score by about 0.44% over SFT.
- The selected Flow-GRPO 50 checkpoint improved by about 0.51% over SFT
  (`+0.002288 ± 0.000787`, 14/16 positive seeds).
- Flow-GRPO added `+0.000090 ± 0.000148` over AWM 800, which is too small to
  claim a stable independent gain.

These are automatic proxy scores; final quality should also be assessed with
blind listening tests.
