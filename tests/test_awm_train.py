"""Small checks for the reward direction in advantage-weighted flow matching."""

from copy import deepcopy
import unittest

import torch
from torch import nn
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from awm_train import group_advantages, matching_losses


class TinyAdaptor(nn.Module):
    def forward(self, conditions):
        return {"c": torch.zeros(len(conditions["id"]), 1)}


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, time, state, embeddings):
        return self.scale * state + embeddings["c"][:, :, None]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TinyBase()
        self.adaptor = TinyAdaptor()


class AdvantageWeightedMatchingTests(unittest.TestCase):
    def test_better_candidate_gets_positive_weight(self):
        scores = [
            {"clap_condition": 0.8, "content_enjoyment": 0.7,
             "production_quality": 0.8},
            {"clap_condition": 0.1, "content_enjoyment": 0.3,
             "production_quality": 0.2},
        ]
        weights = {"clap_condition": 0.5, "content_enjoyment": 0.2,
                   "production_quality": 0.3}
        result = group_advantages(scores, weights, 0.02, "cpu")
        self.assertGreater(result[0].item(), 0)
        self.assertLess(result[1].item(), 0)
        self.assertAlmostEqual(result.sum().item(), 0.0)

    def test_reward_update_reduces_signed_flow_loss(self):
        model = TinyModel()
        reference = deepcopy(model)
        latents = torch.tensor([[[1.0, 1.5]], [[-1.0, -1.5]]])
        ids = torch.tensor([0, 0])
        advantages = torch.tensor([1.0, -1.0])
        matcher = ConditionalFlowMatcher(sigma=0.0)
        torch.manual_seed(42)
        before, anchor, _ = matching_losses(
            model, reference, ids, latents, advantages, matcher
        )
        self.assertAlmostEqual(anchor.item(), 0.0)
        before.backward()
        self.assertIsNotNone(model.base.scale.grad)
        self.assertGreater(abs(model.base.scale.grad.item()), 0)
        with torch.no_grad():
            model.base.scale -= 0.01 * model.base.scale.grad
        torch.manual_seed(42)
        after, anchor, _ = matching_losses(
            model, reference, ids, latents, advantages, matcher
        )
        self.assertLess(after.item(), before.item())
        self.assertGreater(anchor.item(), 0)


if __name__ == "__main__":
    unittest.main()
