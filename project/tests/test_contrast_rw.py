"""CPU tests of actual trainer methods, without loading Transformers/TRL or weights."""
import ast
from pathlib import Path
import types
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[1] / 'src/r1-v/src/open_r1/trainer/grpo_trainer.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen2VLGRPOTrainer')
names = {'_get_per_token_logps', '_score_fixed_completions', '_contrast_token_weights', '_apply_token_rw'}
namespace = {'torch': torch}
for node in trainer.body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
Harness = type('Harness', (), {name: namespace[name] for name in names})


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.4))
        self.dropout = torch.nn.Dropout(0.5)
        self.calls = []

    def generate(self, *args, **kwargs):
        raise AssertionError('Scoring must never generate a third answer')

    def forward(self, ids, attention_mask, pixel_values_videos, use_cache=False):
        self.calls.append((ids.clone(), attention_mask.clone(), self.training))
        vocabulary = torch.arange(12, dtype=torch.float32)
        # Causal, context-sensitive logits; view changes the target distribution.
        context = (ids * attention_mask).cumsum(dim=1).float() % 12
        context = context + pixel_values_videos.reshape(1, 1)
        logits = -(vocabulary - context.unsqueeze(-1)).square() * self.scale
        return types.SimpleNamespace(logits=logits)


class ContrastTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.h.fut_token_sensitivity_tau = 1.0
        self.h.fut_token_max_weight = 1.3
        self.model = TinyPolicy()
        self.a = {'input_ids': torch.tensor([[1, 2]]),
                  'attention_mask': torch.ones(1, 2, dtype=torch.long),
                  'pixel_values_videos': torch.tensor([0.0])}
        self.b = {**self.a, 'pixel_values_videos': torch.tensor([2.0])}
        self.ids = torch.tensor([[3, 4, 5], [6, 7, 0]])
        self.mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    def test_same_tokens_prefix_alignment_and_no_generation(self):
        w = self.h._contrast_token_weights(self.model, self.a, self.b, self.ids, self.mask)
        self.assertEqual(len(self.model.calls), 4)
        for row in range(2):
            expected = torch.cat((self.a['input_ids'], self.ids[row:row+1]), dim=1)
            self.assertTrue(torch.equal(self.model.calls[row][0], expected))
            self.assertTrue(torch.equal(self.model.calls[row+2][0], expected))
        self.assertFalse(w.requires_grad)
        self.assertTrue(self.model.training)
        self.assertTrue(all(not call[2] for call in self.model.calls))
        self.assertEqual(w[1, 2].item(), 1.0)
        self.assertTrue(((w >= 1) & (w <= 1.3)).all())

    def test_identical_views_and_zero_lambda_are_neutral(self):
        w = self.h._contrast_token_weights(self.model, self.a, self.a, self.ids, self.mask)
        self.assertTrue(torch.equal(w, torch.ones_like(w)))
        self.h.fut_token_max_weight = 1.0
        w = self.h._contrast_token_weights(self.model, self.a, self.b, self.ids, self.mask)
        self.assertTrue(torch.equal(w, torch.ones_like(w)))

    def test_scores_predict_target_from_previous_position(self):
        actual = self.h._score_fixed_completions(self.model, self.a, self.ids[:1], self.mask[:1])
        full = torch.cat((self.a['input_ids'], self.ids[:1]), dim=1)
        logits = self.model(full, torch.ones_like(full), self.a['pixel_values_videos']).logits
        expected = logits[:, 1:4].log_softmax(-1).gather(-1, self.ids[:1].unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(actual, expected)

    def test_signed_advantage_and_weight_normalization(self):
        w = self.h._contrast_token_weights(self.model, self.a, self.b, self.ids, self.mask)
        scores = torch.zeros_like(w, requires_grad=True)
        advantages = torch.tensor([1.0, -1.0])
        weighted = self.mask * w
        loss = (-(scores - scores.detach()).exp() * advantages[:, None] * weighted).sum(1) / weighted.sum(1)
        loss.mean().backward()
        self.assertTrue((scores.grad[0] < 0).all())
        self.assertTrue((scores.grad[1, :2] > 0).all())
        self.assertEqual(scores.grad[1, 2].item(), 0.0)
        self.assertIsNone(self.model.scale.grad)
        torch.testing.assert_close(scores.grad.abs().sum(1), torch.tensor([0.5, 0.5]))

    def test_mode_restored_on_failure(self):
        self.model.dropout.eval()
        bad = {**self.b, 'pixel_values_videos': torch.tensor([float('nan')])}
        with self.assertRaises(FloatingPointError):
            self.h._contrast_token_weights(self.model, self.a, bad, self.ids, self.mask)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.dropout.training)

    def test_gap_gate_independent_of_te_bonus(self):
        self.h.temporal = True
        self.h.fut_token_weighting = True
        self.h.fut_token_weighting_scope = 'selective'
        self.h.fut_grpo = False
        self.assertTrue(self.h._apply_token_rw(True, torch.tensor(0.01)))
        for gap in (0.0, -0.01):
            self.assertFalse(self.h._apply_token_rw(True, torch.tensor(gap)))
        self.assertFalse(self.h._apply_token_rw(False, None))
        self.h.fut_token_weighting_scope = 'all_video'
        self.assertTrue(self.h._apply_token_rw(True, torch.tensor(-0.5)))
        self.h.fut_token_weighting = False
        self.assertFalse(self.h._apply_token_rw(True, torch.tensor(1.0)))


if __name__ == '__main__':
    unittest.main()
