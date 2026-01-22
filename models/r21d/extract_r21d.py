from typing import Dict

import numpy as np
import torch

import torchvision
from torchvision.io.video import read_video
import torchvision.models as models

from models._base.base_extractor import BaseExtractor
from models.transforms import (CenterCrop, Normalize, Resize,
                               ToFloatTensorInZeroOne)
from utils.io import reencode_video_with_diff_fps
from utils.utils import form_slices, show_predictions_on_dataset


def _load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        for key in ['state_dict', 'model_state_dict', 'module', 'model']:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return obj


def _clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    strip_prefixes = ['module.', 'model.', 'encoder.', 'backbone.', 'network.']
    drop_prefixes = ['fc.', 'head.', 'classifier.', 'heads.']
    cleaned = {}
    for k, v in state.items():
        if any(k.startswith(dp) for dp in drop_prefixes):
            continue
        for sp in strip_prefixes:
            if k.startswith(sp):
                k = k[len(sp):]
                break
        cleaned[k] = v
    return cleaned


class ExtractR21D(BaseExtractor):

    def __init__(self, args) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
        )
        # (Re-)Define arguments for this class
        r21d_model_cfgs = {
            'r2plus1d_18_16_kinetics': {
                'repo': None,
                'stack_size': 16, 'step_size': 16, 'num_classes': 400, 'dataset': 'kinetics'
            },
            'r2plus1d_34_32_ig65m_ft_kinetics': {
                'repo': 'moabitcoin/ig65m-pytorch', 'model_name_in_repo': 'r2plus1d_34_32_kinetics',
                'stack_size': 32, 'step_size': 32, 'num_classes': 400, 'dataset': 'kinetics'
            },
            'r2plus1d_34_8_ig65m_ft_kinetics': {
                'repo': 'moabitcoin/ig65m-pytorch', 'model_name_in_repo': 'r2plus1d_34_8_kinetics',
                'stack_size': 8, 'step_size': 8, 'num_classes': 400, 'dataset': 'kinetics'
            },
        }
        self.model_name = args.model_name
        self.model_def = r21d_model_cfgs[self.model_name]
        self.extraction_fps = args.extraction_fps
        self.step_size = args.step_size
        self.stack_size = args.stack_size
        if self.step_size is None:
            self.step_size = self.model_def['step_size']
        if self.stack_size is None:
            self.stack_size = self.model_def['stack_size']
        self.show_pred = args.show_pred
        self.checkpoint_path = getattr(args, 'checkpoint_path', None)
        self.auto_convert_checkpoint = getattr(args, 'auto_convert_checkpoint', True)
        self.output_feat_keys = [self.feature_type]
        self.name2module = self.load_model()

    @torch.no_grad()
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        """Extracts features for a given video path.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
        """
        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # read a video
        rgb, audio, info = read_video(video_path, pts_unit='sec')
        # prepare data (first -- transform, then -- unsqueeze)
        rgb = self.transforms(rgb) # could run out of memory here
        rgb = rgb.unsqueeze(0)
        # slice the stack of frames
        slices = form_slices(rgb.size(2), self.stack_size, self.step_size)

        vid_feats = []

        for stack_idx, (start_idx, end_idx) in enumerate(slices):
            # inference
            output = self.name2module['model'](rgb[:, :, start_idx:end_idx, :, :].to(self.device))
            vid_feats.extend(output.tolist())
            self.maybe_show_pred(output, start_idx, end_idx)

        feats_dict = {
            self.feature_type: np.array(vid_feats),
        }

        return feats_dict

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.

        Raises:
            NotImplementedError: if a model is not implemented.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        self.transforms = torchvision.transforms.Compose([
            ToFloatTensorInZeroOne(),
            Resize((128, 171)),
            Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            CenterCrop((112, 112)),
        ])

        if self.model_name == 'r2plus1d_18_16_kinetics':
            weights_key = 'DEFAULT'
            model = models.get_model('r2plus1d_18', weights=None if self.checkpoint_path else weights_key)
        else:
            model = torch.hub.load(
                self.model_def['repo'],
                model=self.model_def['model_name_in_repo'],
                num_classes=self.model_def['num_classes'],
                pretrained=False if self.checkpoint_path else True,
            )

        if self.checkpoint_path:
            raw_state = _load_checkpoint_state(self.checkpoint_path)
            state = _clean_state_dict(raw_state) if self.auto_convert_checkpoint else raw_state
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[r21d] missing keys from checkpoint: {missing}")
            if unexpected:
                print(f"[r21d] unexpected keys in checkpoint: {unexpected}")

        model = model.to(self.device)
        model.eval()
        # save the pre-trained classifier for show_preds and replace it in the net with identity
        class_head = model.fc
        model.fc = torch.nn.Identity()

        return {
            'model': model,
            'class_head': class_head,
        }

    def maybe_show_pred(self, visual_feats: torch.Tensor, start_idx: int, end_idx: int):
        if self.show_pred:
            logits = self.name2module['class_head'](visual_feats)
            print(f'At frames ({start_idx}, {end_idx})')
            show_predictions_on_dataset(logits, self.model_def['dataset'])
