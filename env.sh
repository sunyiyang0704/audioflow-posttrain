#!/usr/bin/env bash
set -euo pipefail

# Install the versions used by the verified Python 3.10 / PyTorch 2.8 setup.
# Repeating the torch pins prevents transitive dependencies from replacing them.
python -m pip install \
  pip==24.0 wheel setuptools==80.9.0 numpy==1.26.4

python -m pip install \
  torch==2.8.0 torchaudio==2.8.0 \
  librosa==0.11.0 torchdiffeq==0.2.5 tqdm==4.67.1 \
  einops==0.8.1 matplotlib==3.10.1

# pypesq is an old package and needs this build workaround on current systems.
CFLAGS=-fcommon python -m pip install \
  --no-build-isolation --no-use-pep517 pypesq==1.2.4

python -m pip install \
  torch==2.8.0 torchaudio==2.8.0 \
  torchcfm==1.0.7 bigvgan descript-audio-codec==1.0.0 \
  audidata==0.0.5 wandb==0.19.10 accelerate==1.10.0 \
  h5py==3.14.0 'transformers>=4.46' audiobox_aesthetics==0.0.4

# stable-audio-tools 0.0.19 pins pandas 2.0.2, while torchcfm's package
# metadata asks for a newer pandas. AudioFlow's runtime imports were verified
# with the version selected by stable-audio-tools.
python -m pip install \
  torch==2.8.0 torchaudio==2.8.0 \
  stable-audio-tools==0.0.19

python -m pip install setuptools==80.9.0
