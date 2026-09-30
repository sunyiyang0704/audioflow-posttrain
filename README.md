# AudioFlow Post-training

This fork reproduces AudioFlow's GTZAN text-to-music baseline and adds
resumable training, LoRA SFT, GROW, and Flow-GRPO experiments. The original
AudioFlow project is available at
[qiuqiangkong/audioflow](https://github.com/qiuqiangkong/audioflow).

## Setup

```bash
conda create -n audioflow python=3.10 -y
conda activate audioflow
bash env.sh
```

The VAE is loaded from `lglg666/SongGeneration-Runtime` by default. Set
`AUDIOFLOW_VAE_REPO_ID` to use another compatible repository.

## GTZAN baseline

```bash
bash scripts/download_gtzan.sh

CUDA_VISIBLE_DEVICES=0 python -m compute_latents.gtzan \
  --dataset_root ./datasets/gtzan \
  --out_dir ./datasets/gtzan_vae \
  --augmentation_repeats 10

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/text2music.yaml --no_log
```

`latest_train.pt` contains the model, EMA, optimizer, scheduler, and completed
step. Set `train.resume_ckpt_path` in the YAML file to resume an interrupted
run. The `step=*_ema.pt` files contain inference weights only.

Generate samples with:

```bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
  --config configs/text2music.yaml \
  --ckpt_path checkpoints/train/text2music/step=500000_ema.pt \
  --out_dir results/text2music
```

Other tasks and datasets are available under `configs/`, `compute_latents/`,
and `scripts/`.

## Post-training

The post-training experiments are split into three scripts:

1. `sft_train.py`: rank-16 LoRA SFT with 0.1 condition dropout;
2. `grow_train.py`: group-relative reward optimization with real-data replay;
3. `flow_grpo_train.py`: stochastic Flow-GRPO with PPO clipping, reference KL,
   waveform verification, and real-data replay.

```bash
python sft_train.py \
  --config configs/text2music_sft.yaml --checkpoint "$BASE_EMA"

python grow_train.py \
  --config configs/text2music_grow.yaml --checkpoint "$SFT_CKPT"

python flow_grpo_train.py \
  --config configs/text2music_flow_grpo.yaml --checkpoint "$GROW_CKPT"
```

Use `evaluate_posttrain.py` for paired fixed-noise evaluation and
`sample_compare.py` to export matched listening samples. See
[`docs/CODE_CHANGES.md`](docs/CODE_CHANGES.md) for the implementation summary
and measured results.

## References

- [Conditional Flow Matching](https://github.com/atong01/conditional-flow-matching)
- [DiT](https://github.com/facebookresearch/DiT)
- [Flow-GRPO](https://arxiv.org/abs/2505.05470)
