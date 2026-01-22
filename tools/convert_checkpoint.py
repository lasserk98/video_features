import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch


DEFAULT_STRIP_PREFIXES = ['module.', 'model.', 'encoder.', 'backbone.', 'network.']
DEFAULT_DROP_PREFIXES = ['fc.', 'head.', 'classifier.', 'heads.']


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        for key in ['state_dict', 'model_state_dict', 'module', 'model']:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return obj


def strip_prefixes(key: str, prefixes: Iterable[str]) -> str:
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key


def clean_state_dict(
    state: Dict[str, torch.Tensor],
    strip_prefixes_list: List[str],
    drop_prefixes_list: List[str],
    keep_classifier: bool,
) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state.items():
        if not keep_classifier and any(k.startswith(dp) for dp in drop_prefixes_list):
            continue
        new_key = strip_prefixes(k, strip_prefixes_list)
        cleaned[new_key] = v
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert checkpoints to plain state_dicts compatible with video_features models.")
    parser.add_argument('--in', dest='src', required=True, type=Path, help='Input checkpoint path (.pth/.pt/.ckpt)')
    parser.add_argument('--out', dest='dst', required=True, type=Path, help='Output checkpoint path (.pth)')
    parser.add_argument('--model', default='resnet', choices=['resnet', 'timm', 'generic'], help='Target model family (controls defaults)')
    parser.add_argument('--strip-prefix', dest='strip_prefix', action='append', default=None,
                        help='Prefixes to strip; can be passed multiple times. Defaults depend on model.')
    parser.add_argument('--drop-prefix', dest='drop_prefix', action='append', default=None,
                        help='Prefixes to drop (e.g., heads); can be passed multiple times. Defaults depend on model.')
    parser.add_argument('--keep-classifier', action='store_true', help='Keep classifier/head weights instead of dropping')
    parser.add_argument('--summary', action='store_true', help='Print missing/unexpected keys after cleaning against target backbone (if available)')

    args = parser.parse_args()

    strip_list = args.strip_prefix or DEFAULT_STRIP_PREFIXES
    drop_list = args.drop_prefix or (DEFAULT_DROP_PREFIXES if args.model in ['resnet', 'timm'] else [])

    state = load_state_dict(args.src)
    cleaned = clean_state_dict(state, strip_list, drop_list, keep_classifier=args.keep_classifier)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cleaned, args.dst)

    print(f"Saved cleaned state_dict with {len(cleaned)} tensors to {args.dst}")
    print("Prefixes stripped:", json.dumps(strip_list))
    print("Prefixes dropped:", json.dumps(drop_list if not args.keep_classifier else []))


if __name__ == '__main__':
    main()
