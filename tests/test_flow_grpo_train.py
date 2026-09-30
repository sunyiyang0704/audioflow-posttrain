"""Numerical checks for the stochastic policy used by Flow-GRPO."""

import unittest
from unittest.mock import patch

import torch

from flow_grpo_train import (
    gaussian_log_ratio, group_advantages, score_group, transition_parameters,
)


class FlowGRPOTrainTests(unittest.TestCase):
    def test_transition_matches_noise_to_data_sde(self):
        state = torch.tensor([[[1.0]]])
        velocity = torch.zeros_like(state)
        mean, variance = transition_parameters(state, velocity, 0.5, 0.1, 0.2)
        self.assertAlmostEqual(mean.item(), 0.996, places=6)
        self.assertAlmostEqual(variance, 0.004, places=6)

    def test_log_ratio_is_gaussian_probability_ratio(self):
        action = torch.tensor([[[1.0]]])
        old_mean = torch.zeros_like(action)
        new_mean = torch.tensor([[[0.1]]], requires_grad=True)
        ratio = gaussian_log_ratio(action, new_mean, old_mean, variance=1.0)
        expected = (
            torch.distributions.Normal(new_mean, 1.0).log_prob(action)
            - torch.distributions.Normal(old_mean, 1.0).log_prob(action)
        ).sum()
        self.assertAlmostEqual(ratio.item(), expected.item(), places=6)
        ratio.sum().backward()
        self.assertGreater(new_mean.grad.item(), 0)

    def test_group_advantage_centers_reward(self):
        advantages = group_advantages(torch.tensor([0.2, 0.5, 0.8]))
        self.assertAlmostEqual(advantages.mean().item(), 0.0, places=6)
        self.assertLess(advantages[0].item(), 0)
        self.assertGreater(advantages[-1].item(), 0)

    def test_verifier_floor_penalizes_only_failing_waveforms(self):
        learned = {
            "total": 0.5, "clap_condition": 0.3,
            "content_enjoyment": 0.6, "production_quality": 0.7,
        }
        with patch("flow_grpo_train.score_rewards", return_value=[learned]), patch(
            "flow_grpo_train.signal_rewards", return_value={"total": 0.7}
        ):
            row = score_group(
                None, [None], 32000, 0, 0.3,
                reward_mode="total_with_verifier_floor",
                verifier_floor=0.78, verifier_penalty=0.4,
            )[0]
        self.assertAlmostEqual(row["hybrid_reward"], 0.468, places=6)


if __name__ == "__main__":
    unittest.main()
