from __future__ import annotations

import importlib
from typing import Callable, Iterable

import numpy as np


GTZAN_LABELS = [
    "blues",
    "classical",
    "country",
    "disco",
    "hip hop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def signal_rewards(audio: np.ndarray, sample_rate: int, condition_id: int) -> dict[str, float]:
    """Detect obvious silence, clipping and flat dynamics.

    These measurements are only used as a penalty around the learned rewards.
    """

    del sample_rate, condition_id
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=0)
    mono = np.nan_to_num(mono)
    if mono.size == 0:
        return {"activity": 0.0, "headroom": 0.0, "dynamics": 0.0, "total": 0.0}

    rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
    clipping = float(np.mean(np.abs(mono) >= 0.995))
    frame = max(256, min(4096, mono.size // 20))
    usable = mono[: mono.size // frame * frame]
    frame_rms = (
        np.sqrt(np.mean(usable.reshape(-1, frame) ** 2, axis=1) + 1e-12)
        if usable.size
        else np.array([rms])
    )
    activity = float(np.clip(rms / 0.08, 0.0, 1.0))
    headroom = float(1.0 - np.clip(clipping / 0.01, 0.0, 1.0))
    dynamics = float(np.clip(np.std(frame_rms) / (np.mean(frame_rms) + 1e-6), 0.0, 1.0))
    total = 0.4 * activity + 0.4 * headroom + 0.2 * dynamics
    return {
        "activity": activity,
        "headroom": headroom,
        "dynamics": dynamics,
        "total": float(total),
    }


class PretrainedAudioReward:
    """CLAP condition alignment plus Audiobox learned aesthetic quality."""

    def __init__(self, config: dict):
        import torch
        from audiobox_aesthetics.infer import initialize_predictor
        from transformers import AutoProcessor, ClapModel

        self.torch = torch
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("reward.device is cuda, but CUDA is unavailable")

        self.labels = config.get("labels", GTZAN_LABELS)
        self.prompts = config.get(
            "prompts", [f"a high-quality {label} music recording" for label in self.labels]
        )
        if len(self.prompts) != len(self.labels):
            raise ValueError("reward prompts and labels must have the same length")

        clap_id = config.get("clap_model", "laion/clap-htsat-unfused")
        print(f"Loading CLAP reward model: {clap_id}")
        self.clap_processor = AutoProcessor.from_pretrained(clap_id)
        self.clap = ClapModel.from_pretrained(clap_id).to(self.device).eval()
        self.clap_sr = int(self.clap_processor.feature_extractor.sampling_rate)

        print("Loading Audiobox Aesthetics reward model: facebook/audiobox-aesthetics")
        self.aesthetics = initialize_predictor(config.get("audiobox_checkpoint"))
        self.weights = {
            "clap": 0.5,
            "content_enjoyment": 0.2,
            "production_quality": 0.3,
            "signal_penalty": 0.1,
            **config.get("weights", {}),
        }

    @staticmethod
    def _mono(audio: np.ndarray) -> np.ndarray:
        array = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        return array.mean(axis=0) if array.ndim == 2 else array

    def score_batch(
        self,
        audios: Iterable[np.ndarray],
        sample_rate: int,
        condition_ids: Iterable[int],
    ) -> list[dict[str, float]]:
        import torchaudio

        audios = list(audios)
        condition_ids = [int(item) for item in condition_ids]
        if len(audios) != len(condition_ids):
            raise ValueError("audios and condition_ids must have the same length")

        clap_audio = []
        aesthetics_input = []
        for audio in audios:
            mono = self.torch.from_numpy(self._mono(audio)).float().unsqueeze(0)
            aesthetics_input.append({"path": mono, "sample_rate": sample_rate})
            if sample_rate != self.clap_sr:
                mono = torchaudio.functional.resample(mono, sample_rate, self.clap_sr)
            clap_audio.append(mono.squeeze(0).numpy())

        inputs = self.clap_processor(
            text=self.prompts,
            audio=clap_audio,
            sampling_rate=self.clap_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.clap(**inputs).logits_per_audio
            genre_probabilities = logits.softmax(dim=-1).cpu().numpy()
        aesthetic_scores = self.aesthetics.forward(aesthetics_input)

        results = []
        for index, (audio, condition_id, axes) in enumerate(
            zip(audios, condition_ids, aesthetic_scores)
        ):
            if not 0 <= condition_id < len(self.labels):
                raise ValueError(f"condition_id {condition_id} is outside reward labels")
            clap_score = float(genre_probabilities[index, condition_id])
            enjoyment = float(np.clip((axes["CE"] - 1.0) / 9.0, 0.0, 1.0))
            production_quality = float(np.clip((axes["PQ"] - 1.0) / 9.0, 0.0, 1.0))
            signal = signal_rewards(audio, sample_rate, condition_id)
            total = (
                self.weights["clap"] * clap_score
                + self.weights["content_enjoyment"] * enjoyment
                + self.weights["production_quality"] * production_quality
                - self.weights["signal_penalty"] * (1.0 - signal["total"])
            )
            results.append(
                {
                    "clap_condition": clap_score,
                    "content_enjoyment": enjoyment,
                    "production_quality": production_quality,
                    "audiobox_CE_raw": float(axes["CE"]),
                    "audiobox_PQ_raw": float(axes["PQ"]),
                    "signal_quality": float(signal["total"]),
                    "total": float(total),
                }
            )
        return results

    def __call__(self, audio: np.ndarray, sample_rate: int, condition_id: int):
        return self.score_batch([audio], sample_rate, [condition_id])[0]


def load_reward_function(config: dict | None = None, spec: str | None = None) -> Callable:
    """Build the learned reward ensemble, or load a custom ``module:function``."""

    if spec:
        module_name, function_name = spec.split(":", maxsplit=1)
        return getattr(importlib.import_module(module_name), function_name)
    config = config or {}
    if config.get("backend", "pretrained") == "signal":
        return signal_rewards
    return PretrainedAudioReward(config)


def score_rewards(reward_fn, audios, sample_rate: int, condition_ids):
    """Use batched model inference when the reward implementation supports it."""

    if hasattr(reward_fn, "score_batch"):
        return reward_fn.score_batch(audios, sample_rate, condition_ids)
    return [
        reward_fn(audio, sample_rate, int(condition_id))
        for audio, condition_id in zip(audios, condition_ids)
    ]
