"""Run in a short-lived process so CUDA inspection leaves no launcher GPU context."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from launch_3b import ROOT, TRAIN, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'server_3b_config.json')
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_visible_devices']
    for path in reversed([ROOT / 'project/src/qwen-vl-utils/src', TRAIN / 'src', TRAIN / 'src/open_r1']):
        sys.path.insert(0, str(path))
    if sys.version_info < (3, 10):
        raise RuntimeError('Use the existing Python >=3.10 videor1 server environment')
    for key in ['model_path', 'data_root', 'dataset_path']:
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(key + ': ' + cfg[key])
    config = json.loads((Path(cfg['model_path']) / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'qwen2_5_vl':
        raise ValueError('Expected Qwen2.5-VL SFT model, not an unrelated backbone')
    hidden = config.get('text_config', config).get('hidden_size')
    if hidden and hidden != 2048:
        raise ValueError(f'Expected 3B text hidden_size=2048, got {hidden}; verify SFT model path')
    model = Path(cfg['model_path'])
    indices = list(model.glob('*.index.json'))
    weight_indices = [p for p in indices if 'safetensors' in p.name or 'pytorch_model' in p.name]
    if weight_indices:
        for index in weight_indices:
            mapping = json.loads(index.read_text(encoding='utf-8'))['weight_map']
            for shard in set(mapping.values()):
                if not (model / shard).is_file():
                    raise FileNotFoundError('Missing model shard: ' + str(model / shard))
    elif not list(model.glob('*.safetensors')) and not (model / 'pytorch_model.bin').is_file():
        raise FileNotFoundError('SFT model config exists but model weights were not found')
    if not (model / 'preprocessor_config.json').is_file():
        raise FileNotFoundError('SFT directory lacks preprocessor_config.json; use the complete SFT export')
    versions = {}
    for module in ['torch', 'torchvision', 'transformers', 'accelerate', 'trl', 'deepspeed',
                   'datasets', 'nltk', 'rouge_score', 'qwen_vl_utils']:
        obj = importlib.import_module(module)
        versions[module] = getattr(obj, '__version__', 'installed')
    import torch
    import qwen_vl_utils
    if str(ROOT / 'project/src/qwen-vl-utils/src') not in str(Path(qwen_vl_utils.__file__).resolve()):
        raise RuntimeError('Wrong qwen_vl_utils imported; local 16-frame sampler must take precedence')
    from transformers import Qwen2_5_VLForConditionalGeneration
    from trl import GRPOConfig
    for key in ['use_vllm', 'model_init_kwargs']:
        if key not in GRPOConfig.__dataclass_fields__:
            raise RuntimeError('TRL GRPOConfig API differs: missing ' + key + '; this trainer was based on TRL 0.16.0')
    from trainer import Qwen2VLGRPOTrainer
    backend = os.environ.get('FORCE_QWENVL_VIDEO_READER')
    if not backend:
        backend = 'decord' if importlib.util.find_spec('decord') else 'torchvision'
    if backend == 'decord':
        importlib.import_module('decord')
    elif backend == 'torchvision':
        importlib.import_module('av')
    else:
        raise ValueError('Unsupported FORCE_QWENVL_VIDEO_READER: ' + backend)
    if cfg['attn_implementation'] == 'flash_attention_2':
        importlib.import_module('flash_attn')
    if not torch.cuda.is_available() or torch.cuda.device_count() != cfg['world_size']:
        raise RuntimeError('Visible CUDA GPU count differs from CUDA_VISIBLE_DEVICES')
    devices = []
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        devices.append({'logical_index': index, 'name': prop.name, 'memory_gib': round(prop.total_memory / 2**30, 1)})
    existing = Path(cfg['output_root'])
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free / 2**30
    if free < 100:
        print(f'WARNING: only {free:.1f} GiB free; full optimizer-state checkpoints can be large')
    print(json.dumps({'preflight': 'PASSED', 'python': sys.version, 'versions': versions,
                      'visible_devices': devices, 'free_disk_gib': round(free, 1),
                      'qwen_vl_utils': qwen_vl_utils.__file__, 'text_hidden_size': hidden,
                      'training_max_frames': 16, 'video_reader': backend, 'vllm': False}, indent=2))


if __name__ == '__main__':
    main()
