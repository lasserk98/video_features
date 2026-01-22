from typing import Dict
import omegaconf

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from models._base.base_framewise_extractor import BaseFrameWiseExtractor
from utils.utils import show_predictions_on_dataset


def _load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and return a state_dict, handling Lightning wrappers."""
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        for key in ['state_dict', 'model_state_dict', 'module', 'model']:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return obj


def _clean_state_dict(state: Dict[str, torch.Tensor], *, strip_prefixes=None, drop_prefixes=None) -> Dict[str, torch.Tensor]:
    """Strip common prefixes and drop classifier/head weights."""
    strip_prefixes = strip_prefixes or ['module.', 'model.', 'encoder.', 'backbone.', 'network.']
    drop_prefixes = drop_prefixes or ['fc.', 'head.', 'classifier.', 'heads.']

    cleaned = {}
    for k, v in state.items():
        # drop unwanted heads
        if any(k.startswith(dp) for dp in drop_prefixes):
            continue
        # strip known prefixes
        for sp in strip_prefixes:
            if k.startswith(sp):
                k = k[len(sp):]
                break
        cleaned[k] = v
    return cleaned


class ExtractResNet(BaseFrameWiseExtractor):

    def __init__(self, args: omegaconf.DictConfig) -> None:
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
            model_name=args.model_name,
            batch_size=args.batch_size,
            extraction_fps=args.extraction_fps,
            extraction_total=args.extraction_total,
            show_pred=args.show_pred,
            use_amp=getattr(args, 'use_amp', False),
        )
        self.checkpoint_path = args.get('checkpoint_path', None)
        self.auto_convert_checkpoint = args.get('auto_convert_checkpoint', True)
        self.name2module = self.load_model()

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.

        Raises:
            NotImplementedError: if a model is not implemented.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        weights_key = 'IMAGENET1K_V1'
        weights = models.get_model_weights(self.model_name)[weights_key]
        use_pretrained = self.checkpoint_path is None
        model = models.get_model(self.model_name, weights=weights_key if use_pretrained else None)

        if self.checkpoint_path:
            raw_state = _load_checkpoint_state(self.checkpoint_path)
            state = _clean_state_dict(raw_state) if self.auto_convert_checkpoint else raw_state
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[resnet] missing keys from checkpoint: {missing}")
            if unexpected:
                print(f"[resnet] unexpected keys in checkpoint: {unexpected}")

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            weights.transforms(),
        ])

        model = model.to(self.device)
        model.eval()
        # save the pre-trained classifier for show_preds and replace it in the net with identity
        class_head = model.fc
        model.fc = torch.nn.Identity()
        return {
            'model': model,
            'class_head': class_head,
        }

    def maybe_show_pred(self, feats: torch.Tensor):
        if self.show_pred:
            logits = self.name2module['class_head'](feats)
            show_predictions_on_dataset(logits, 'imagenet1k')
