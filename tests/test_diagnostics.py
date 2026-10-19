import ast
import json
import os
from pathlib import Path
import runpy
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
base = runpy.run_path(str(ROOT / 'project/tests/test_contrast_rw.py'), run_name='base_tests')
source = ROOT / 'project/src/r1-v/src/open_r1/trainer/grpo_trainer.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen2VLGRPOTrainer')
node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_record_rw_diagnostics')
namespace = {'torch': torch, 'json': json, 'os': os}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.base = base['ContrastTests']()
        self.base.setUp()

    def test_prompt_mismatch_rejected(self):
        c = self.base
        bad = {**c.b, 'input_ids': torch.tensor([[1, 9]])}
        with self.assertRaises(ValueError):
            c.h._contrast_token_weights(c.model, c.a, bad, c.ids, c.mask)

    def test_inactive_unit_weights_and_trace_values(self):
        c = self.base
        w = c.h._contrast_token_weights(c.model, c.a, c.b, c.ids, c.mask)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
                'TE_RW_DIAGNOSTICS_EVERY': '1', 'TE_RW_TRACE_EVERY': '1'}, clear=False):
            c.h.accelerator = types.SimpleNamespace(process_index=0)
            c.h.args = types.SimpleNamespace(output_dir=temp)
            c.h.state = types.SimpleNamespace(global_step=0)
            record = namespace['_record_rw_diagnostics']
            record(c.h, w, c.ids, c.mask, torch.tensor([1., -1.]), torch.tensor(-0.2), True, False, {'problem_id': 1})
            record(c.h, w, c.ids, c.mask, torch.tensor([1., -1.]), torch.tensor(0.2), True, True, {'problem_id': 1})
            rows = [json.loads(x) for x in (Path(temp) / 'rw_diagnostics.rank0.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['applied_weight_mean'], 1.0)
            self.assertEqual(rows[0]['relative_weight_std'], 0.0)
            self.assertGreater(rows[1]['relative_weight_std'], 0.0)
            traces = [json.loads(x) for x in (Path(temp) / 'rw_tokens.rank0.jsonl').read_text().splitlines()]
            self.assertEqual(traces[0]['completion_token_ids'], c.ids[0].tolist())
            self.assertEqual(traces[0]['applied_weights'], [1., 1., 1.])
            self.assertEqual(len(traces[1]['logp_ordered']), 3)

    def test_no_visual_fallback_and_no_eager_vllm(self):
        text = source.read_text(encoding='utf-8')
        self.assertIn('refusing to retry without visual inputs', text)
        self.assertNotIn('self._get_per_token_logps(model, prompt_completion_ids)', text)
        entry = (ROOT / 'project/src/r1-v/src/open_r1/grpo.py').read_text(encoding='utf-8')
        self.assertNotIn('from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified', entry)


if __name__ == '__main__':
    unittest.main()
