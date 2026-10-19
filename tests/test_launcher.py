import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('server_launcher', ROOT / 'launch_3b.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LauncherTests(unittest.TestCase):
    def config(self):
        with patch.dict(os.environ, {}, clear=True):
            return launch.load_config(ROOT / 'server_3b_config.json')

    def args(self, variant, smoke=False):
        command, settings = launch.build_command(self.config(), variant, '/tmp/new_run', '/data/full.json', smoke)
        return command, settings

    def test_selective_explicit_probability_recipe(self):
        command, s = self.args('selective')
        self.assertFalse(s['use_vllm'])
        self.assertTrue(s['fut_token_weighting'])
        self.assertEqual(s['fut_token_weighting_method'], 'contrast')
        self.assertEqual(s['fut_token_weighting_scope'], 'selective')
        self.assertEqual((s['fut_temporal_alpha'], s['fut_temporal_margin']), (0.5, 0.25))
        self.assertEqual((s['num_generations'], s['fut_negative_generations']), (2, 1))
        self.assertEqual(s['fut_gap_threshold'], 0.0)
        self.assertEqual(s['fut_min_correct_reward'], 0.0)
        self.assertEqual(command[-3:], ['--reward_funcs', 'soft_accuracy', 'format'])

    def test_baseline_both_switches_disabled(self):
        command, s = self.args('baseline')
        self.assertFalse(s['fut_grpo'])
        self.assertFalse(s['fut_token_weighting'])
        self.assertEqual(command[-3:], ['--reward_funcs', 'accuracy', 'format'])

    def test_scope_ablation_changes_only_scope(self):
        _, a = self.args('selective')
        _, b = self.args('all_video')
        self.assertEqual([k for k in a if a[k] != b[k]], ['fut_token_weighting_scope'])

    def test_no_rw_keeps_proposed_reward(self):
        _, s = self.args('no_rw')
        self.assertTrue(s['fut_grpo'])
        self.assertFalse(s['fut_token_weighting'])

    def test_smoke_does_not_save_and_preserves_input_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            _, s = self.args('selective', True)
        self.assertEqual(s['max_steps'], 2)
        self.assertEqual(s['save_strategy'], 'no')
        self.assertEqual((s['max_prompt_length'], s['max_completion_length']), (4096, 192))

    def test_env_override_and_duplicate_gpu_rejection(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,1'}, clear=True):
            with self.assertRaises(ValueError):
                launch.load_config(ROOT / 'server_3b_config.json')
        with patch.dict(os.environ, {'TE_ALPHA': '0.7', 'MODEL_PATH': '/my/sft'}, clear=True):
            cfg = launch.load_config(ROOT / 'server_3b_config.json')
            self.assertEqual(cfg['alpha'], 0.7)
            self.assertEqual(cfg['model_path'], '/my/sft')

    def test_local_sampler_first_and_backend_explicit(self):
        env = launch.job_environment(self.config(), True)
        self.assertTrue(env['PYTHONPATH'].startswith(str(ROOT / 'project/src/qwen-vl-utils/src')))
        self.assertEqual(env['TE_CGRPO_SMOKE'], '1')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,3')

    def test_mixed_smoke_subset_does_not_modify_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for mode in ['video', 'image']:
                media = root / (mode + '.bin')
                media.write_bytes(b'fixture only; not decoded')
                rows.append(dict(problem='q', solution='<answer>A</answer>', data_type=mode,
                                 problem_type='multiple choice', options=['A. one'], path='./' + media.name))
            dataset = root / 'data.json'
            dataset.write_text(json.dumps(rows), encoding='utf-8')
            before = dataset.read_bytes()
            cfg = self.config()
            cfg.update(data_root=str(root), dataset_path=str(dataset))
            meta, subset = launch.inspect_dataset(cfg)
            self.assertEqual(meta['records'], 2)
            self.assertEqual(len(subset), 6)
            self.assertEqual({r['data_type'] for r in subset}, {'video', 'image'})
            self.assertEqual(dataset.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
