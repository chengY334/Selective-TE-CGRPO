#!/usr/bin/env python3
"""Explicit, non-vLLM 3B launcher. No model/data download and no old-run overwrite."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / 'project/src/r1-v'
VARIANTS = {
    'selective': ('soft_accuracy', True, True, 'selective', 'contrast'),
    'all_video': ('soft_accuracy', True, True, 'all_video', 'contrast'),
    'no_rw': ('soft_accuracy', True, False, 'selective', 'contrast'),
    'baseline': ('accuracy', False, False, 'selective', 'contrast'),
    'lexicon': ('soft_accuracy', True, True, 'selective', 'lexicon'),
}
ENV = {
    'MODEL_PATH': ('model_path', str), 'DATA_ROOT': ('data_root', str),
    'DATASET_PATH': ('dataset_path', str), 'OUTPUT_ROOT': ('output_root', str),
    'CUDA_VISIBLE_DEVICES': ('cuda_visible_devices', str), 'SEED': ('seed', int),
    'MAX_STEPS': ('max_steps', int), 'SAVE_STEPS': ('save_steps', int),
    'MAX_PROMPT_LENGTH': ('max_prompt_length', int),
    'MAX_COMPLETION_LENGTH': ('max_completion_length', int),
    'MAX_PIXELS': ('max_pixels', int),
    'GRADIENT_ACCUMULATION_STEPS': ('gradient_accumulation_steps', int),
    'TE_ALPHA': ('alpha', float), 'TE_MARGIN': ('margin', float),
    'RW_MAX_WEIGHT': ('token_max_weight', float), 'RW_TAU': ('token_tau', float),
    'ATTN_IMPLEMENTATION': ('attn_implementation', str),
    'RW_DIAGNOSTICS_EVERY': ('diagnostics_every', int),
    'RW_TRACE_EVERY': ('token_trace_every', int),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for name, (key, cast) in ENV.items():
        if os.environ.get(name):
            cfg[key] = cast(os.environ[name])
    devices = [s.strip() for s in cfg['cuda_visible_devices'].split(',') if s.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError('CUDA_VISIBLE_DEVICES must list unique GPUs, e.g. 0,1,3')
    cfg['cuda_visible_devices'] = ','.join(devices)
    cfg['world_size'] = len(devices)
    if cfg['world_size'] != 3:
        print('WARNING: not 3 training GPUs; effective batch/hardware no longer match the 3B paper recipe.')
    for key in ['max_steps', 'save_steps', 'max_prompt_length', 'max_completion_length',
                'max_pixels', 'gradient_accumulation_steps']:
        if cfg[key] < 1:
            raise ValueError(key + ' must be positive')
    if cfg['token_tau'] <= 0 or cfg['token_max_weight'] < 1 or cfg['margin'] <= 0 or cfg['alpha'] <= 0:
        raise ValueError('Invalid token/TE parameters')
    return cfg


def job_environment(cfg, smoke):
    env = os.environ.copy()
    paths = [ROOT / 'project/src/qwen-vl-utils/src', TRAIN / 'src', TRAIN / 'src/open_r1']
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths)) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(CUDA_VISIBLE_DEVICES=cfg['cuda_visible_devices'], VIDEO_R1_DATA_ROOT=cfg['data_root'],
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED=str(cfg['seed']),
               TE_CGRPO_SMOKE='1' if smoke else '0',
               TE_RW_DIAGNOSTICS_EVERY=str(cfg['diagnostics_every']),
               TE_RW_TRACE_EVERY=str(cfg['token_trace_every']), TE_RW_TRACE_MAX_TOKENS='192')
    env.setdefault('OMP_NUM_THREADS', '8')
    env.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
    env.setdefault('DEBUG_MODE', 'false')
    return env


def data_records(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig') as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        f.seek(0)
        records = json.load(f) if first == '[' else [json.loads(line) for line in f if line.strip()]
    if not isinstance(records, list) or not records:
        raise ValueError('Expected a nonempty JSON array or JSONL dataset')
    return records


def inspect_dataset(cfg):
    records = data_records(cfg['dataset_path'])
    valid_types = {'multiple choice', 'numerical', 'regression', 'OCR', 'free-form'}
    counts, tasks = Counter(), Counter()
    picked = {'image': [], 'video': []}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Dataset records must be objects')
        for key in ['problem', 'solution', 'data_type', 'problem_type', 'path']:
            if key not in row:
                raise ValueError('Dataset record is missing ' + key)
        modality = row['data_type']
        if modality not in picked or row['problem_type'] not in valid_types:
            raise ValueError('Unsupported modality/task: ' + str((modality, row['problem_type'])))
        if row['problem_type'] == 'multiple choice' and not isinstance(row.get('options'), list):
            raise ValueError('Multiple-choice record requires an options list')
        counts[modality] += 1
        tasks[row['problem_type']] += 1
        if len(picked[modality]) < 3:
            media = Path(row['path'])
            if not media.is_absolute():
                media = Path(cfg['data_root']) / row['path'].lstrip('./\\')
            if media.is_file():
                picked[modality].append(row)
    if not picked['video'] or not picked['image']:
        raise FileNotFoundError('Cannot locate both image and video smoke samples under DATA_ROOT; inspect dataset paths')
    sample = picked['video'] + picked['image']
    while len(sample) < 6:
        sample += sample[:6 - len(sample)]
    meta = {'records': len(records), 'modalities': dict(counts), 'tasks': dict(tasks),
            'sha256': sha(cfg['dataset_path']), 'smoke_records': len(sample),
            'note': 'Training uses the full supplied dataset; smoke alone uses a mixed six-record subset.'}
    return meta, sample


def build_command(cfg, variant, output, dataset, smoke=False, resume=None):
    scorer, te, rw, scope, method = VARIANTS[variant]
    steps = int(os.environ.get('SMOKE_STEPS', '2')) if smoke else cfg['max_steps']
    if smoke and not 1 <= steps <= 10:
        raise ValueError('SMOKE_STEPS must be 1..10; use train for longer runs')
    settings = {
        'output_dir': str(output), 'model_name_or_path': cfg['model_path'],
        'dataset_name': str(dataset), 'use_vllm': False,
        'deepspeed': str(TRAIN / 'local_scripts/zero3_cpu_offload.json'),
        'max_prompt_length': cfg['max_prompt_length'], 'max_completion_length': cfg['max_completion_length'],
        'per_device_train_batch_size': 1, 'gradient_accumulation_steps': cfg['gradient_accumulation_steps'],
        'learning_rate': 1e-6, 'lr_scheduler_type': 'cosine', 'weight_decay': 0.01,
        'logging_steps': 1, 'bf16': True, 'gradient_checkpointing': True,
        'attn_implementation': cfg['attn_implementation'], 'min_pixels': 3136, 'max_pixels': cfg['max_pixels'],
        'num_train_epochs': 1, 'max_steps': steps, 'seed': cfg['seed'], 'data_seed': cfg['seed'],
        'run_name': Path(output).name, 'save_strategy': 'no' if smoke else 'steps',
        'save_steps': cfg['save_steps'], 'save_only_model': False,
        'temporal': True, 'len_control': True, 'fut_grpo': te, 'fut_token_weighting': rw,
        'fut_token_weighting_method': method, 'fut_token_weighting_scope': scope,
        'fut_temporal_alpha': cfg['alpha'], 'fut_temporal_margin': cfg['margin'],
        'fut_gap_threshold': 0.0, 'fut_min_correct_reward': 0.0,
        'fut_corruptions': 'shuffle', 'fut_num_corruptions': 1, 'fut_negative_generations': 1,
        'fut_token_max_weight': cfg['token_max_weight'], 'fut_token_sensitivity_tau': cfg['token_tau'],
        'report_to': 'none', 'beta': 0.04, 'max_grad_norm': 5, 'temperature': 1.0, 'num_generations': 2,
    }
    if resume:
        settings['resume_from_checkpoint'] = str(resume)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               '--nproc_per_node=' + str(cfg['world_size']), 'src/open_r1/grpo.py']
    for key, value in settings.items():
        command += ['--' + key, str(value).lower() if isinstance(value, bool) else str(value)]
    command += ['--reward_funcs', scorer, 'format']
    return command, settings


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'project').rglob('*.py'))
            if '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'smoke', 'train'])
    parser.add_argument('--variant', choices=VARIANTS, default='selective')
    parser.add_argument('--config', type=Path, default=ROOT / 'server_3b_config.json')
    parser.add_argument('--dry-run', action='store_true', help='Print command only; no path/dependency checks or training')
    parser.add_argument('--run-name', help='New directory name, not a path')
    parser.add_argument('--resume', type=Path, help='Resume only from this bundle\'s compatible checkpoint')
    args = parser.parse_args()
    cfg = load_config(args.config)
    smoke = args.action == 'smoke'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in {'.', '..'}):
        parser.error('--run-name must be a simple directory name')
    if args.resume and args.action != 'train':
        parser.error('--resume is for train only')
    name = args.run_name or f"{'SMOKE_' if smoke else ''}{args.variant}_3b_seed{cfg['seed']}_{stamp}"
    output = args.resume.resolve().parent if args.resume else Path(cfg['output_root']) / name
    dataset = output / 'smoke_dataset.json' if smoke else Path(cfg['dataset_path'])
    command, settings = build_command(cfg, args.variant, output, dataset, smoke, args.resume)
    print('Variant:', args.variant, '| GPUs:', cfg['cuda_visible_devices'], '| alpha/m:', cfg['alpha'], cfg['margin'], flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if os.name == 'nt':
        parser.error('Training bundle targets your Linux CUDA server; use --dry-run locally')
    env = job_environment(cfg, smoke)
    doctor = subprocess.run([sys.executable, '-B', str(ROOT / 'server_preflight.py'), '--config', str(args.config)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(doctor.stdout, flush=True)
    if doctor.returncode:
        raise SystemExit(doctor.returncode)
    meta, sample = inspect_dataset(cfg)
    print('Dataset:', json.dumps(meta, ensure_ascii=False), flush=True)
    if args.action == 'preflight':
        return
    hashes = source_hashes()
    if args.resume:
        if not (args.resume / 'trainer_state.json').is_file():
            raise FileNotFoundError('Resume checkpoint lacks trainer_state.json')
        old = json.loads((output / 'run_manifest.json').read_text(encoding='utf-8'))
        if old['variant'] != args.variant or old['config'] != cfg or old['source_sha256'] != hashes or old['dataset']['sha256'] != meta['sha256']:
            raise ValueError('Resume refused: variant/config/source/dataset differs from original manifest')
    else:
        output.mkdir(parents=True, exist_ok=False)
    if smoke:
        dataset.write_text(json.dumps(sample, ensure_ascii=False), encoding='utf-8')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'variant': args.variant,
                'smoke_only': smoke, 'config': cfg, 'training_arguments': settings,
                'dataset': meta, 'source_sha256': hashes, 'command': command,
                'model_config_sha256': sha(Path(cfg['model_path']) / 'config.json'),
                'model_identity_note': 'Config and path recorded; model weight bytes are not fully hashed.'}
    if args.resume:
        history = output / 'resume_history'
        history.mkdir(exist_ok=True)
        manifest_file = history / (stamp + '.json')
    else:
        manifest_file = output / 'run_manifest.json'
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / ('environment_' + stamp + '.txt')).write_text(doctor.stdout, encoding='utf-8')
    (output / ('command_' + stamp + '.txt')).write_text(shlex.join(command) + '\n', encoding='utf-8')
    env['LOG_PATH'] = str(output / 'debug.log')
    start = time.monotonic()
    with (output / 'launcher.log').open('a', encoding='utf-8') as log:
        proc = subprocess.Popen(command, cwd=TRAIN, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            raise
    result = {'exit_code': code, 'elapsed_seconds': time.monotonic() - start,
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'smoke_only': smoke}
    (output / (('COMPLETE_' if code == 0 else 'FAILED_') + stamp + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    if code:
        raise SystemExit(code)
    print('Completed:', output, flush=True)


if __name__ == '__main__':
    main()
