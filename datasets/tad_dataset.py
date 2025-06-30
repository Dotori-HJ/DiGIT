# ------------------------------------------------------------------------
# Modified from TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021 - 2022. Xiaolong Liu.
# ------------------------------------------------------------------------

'''Universal TAD Dataset loader.'''

if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import json
import random
import os.path as osp

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset

from .data_util import load_feature, get_classes, truncate_feats, add_noise_to_segments



class TADDataset(Dataset):
    def __init__(self, feature_folder, gt_path, base_frame, stride, subset, mode, ignored_videos, name_format='{}', transforms=None,
                 max_seq_len=None, resize=False, downsample_rate=1.0, normalize=False, mem_cache=True, noise_scale=0, noise_scaler=0, seg_noise_scale=0, binary=False, default_fps=30):
        '''TADDataset
        Parameters:
            subset: train/val/test
            mode: train, or test
            ann_file: path to the ground truth file
            ft_info_file: path to the file that describe other information of each video
            transforms: which transform to use
            mem_cache: cache features of the whole dataset into memory.
            binary: transform all gt to binary classes. This is required for training a class-agnostic detector
            padding: whether to pad the input feature to `slice_len`
        '''

        super().__init__()
        self.feature_folder = feature_folder
        self.gt_path = gt_path
        self.default_fps = default_fps
        self.base_frame = base_frame
        self.stride = stride
        self.subset = subset
        self.mode = mode
        self.ignored_videos = ignored_videos
        self.name_format = name_format
        self.transforms = transforms
        self.max_seq_len = max_seq_len
        self.resize = resize
        self.downsample_rate = downsample_rate
        self.normalize = normalize
        self.mem_cache = mem_cache
        self.binary = binary
        self.noise_scale = noise_scale
        self.noise_scaler = noise_scaler
        self.seg_noise_scale = seg_noise_scale

        with open(gt_path, 'rt') as f:
            self.gt = json.load(f)['database']

        # self.remove_duplicated_and_short()
        self.video_names = []
        for video_name, video_info in self.gt.items():
            # Filtering out 'Ambiguous' annotations
            if video_info['subset'] == subset and not video_name in ignored_videos:
                self.video_names.append(video_name)

        self.classes = get_classes(self.gt)
        self.classname_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_classname = {idx: cls for idx, cls in enumerate(self.classes)}

        self.remove_duplicated_and_short()

        if self.mem_cache:
            self.cache = {}
            for video_name in self.video_names:
                self.cache[video_name] = self._load_feature(video_name)
            
            
    def sort_video_names(self):
        video_lengths = {video_name: float(self.gt[video_name]['duration']) for video_name in self.video_names}
        self.video_names.sort(key=lambda x: video_lengths[x])

    def remove_duplicated_and_short(self, eps=0.02):
        num_removed = 0
        for vid in self.gt.keys():
            annotations = self.gt[vid]['annotations']
            valid_annos = []

            for anno in annotations:
                s, e = anno["segment"]
                l = anno["label"]

                if (e - s) >= eps:
                    valid = True
                else:
                    valid = False
                for v_anno in valid_annos:
                    if ((abs(s - v_anno['segment'][0]) <= eps)
                        and (abs(e - v_anno['segment'][1]) <= eps)
                        and (l == v_anno['label'])
                    ):
                        valid = False
                        break

                if valid:
                    valid_annos.append(anno)
                else:
                    num_removed += 1

            self.gt[vid]['annotations'] = valid_annos
        if num_removed > 0:
            print(f"Removed {num_removed} duplicated and short annotations")

    def __len__(self):
        return len(self.video_names)

    def _load_feature(self, video_name):
        if self.mem_cache and video_name in self.cache:
            return self.cache[video_name]
        feature_path = osp.join(self.feature_folder, self.name_format.format(video_name))
        return load_feature(feature_path)

    def _load_annotations(self, video_name):
        annotations = self.gt[video_name]['annotations']
        segments, labels = [], []
        if len(annotations) == 0:
            segments = torch.empty((0, 2), dtype=torch.float32)
            labels = torch.empty((0, ), dtype=torch.long)
            return segments, labels

        segments = torch.tensor([anno['segment'] for anno in annotations], dtype=torch.float32)
        if self.binary:
            labels = torch.zeros(len(annotations), dtype=torch.long)
        else:
            labels = torch.tensor([
                self.classname_to_idx[anno['label']]
                for anno in annotations
            ], dtype=torch.long)

        # segments = torch.stack(segments)
        # labels = torch.stack(labels)
        return segments, labels

    def __getitem__(self, i):
        video_name = self.video_names[i]
        features = self._load_feature(video_name)
        segments, labels = self._load_annotations(video_name)
        video_duration = float(self.gt[video_name]['duration'])

        base_frames = self.base_frame
        stride = self.stride
        fps = self.default_fps if 'fps' not in self.gt[video_name] else float(self.gt[video_name]['fps'])

        if self.mode == 'train' and self.seg_noise_scale > 0:
            segments = add_noise_to_segments(segments, self.seg_noise_scale)

        if self.downsample_rate > 1:
            if self.mode == 'train':
                st_idx = random.randrange(min(features.size(1), self.downsample_rate))
            else:
                st_idx = 0
            segments = segments - st_idx * (stride / fps)
            features = features[:, st_idx::self.downsample_rate]

            stride = stride * self.downsample_rate

        if stride != base_frames:
            offset = (base_frames - stride) * 0.5 / fps
            segments = segments - offset
        else:
            offset = 0

        if self.resize:
            if self.max_seq_len is not None:
                scale_factor = self.max_seq_len / features.size(1)
                features = F.interpolate(features.unsqueeze(0), self.max_seq_len, mode='linear').squeeze(0)
                fps = fps * scale_factor
            else:
                scale_factor = fps / self.default_fps
                # scale_factor = self.default_fps / fps
                features = F.interpolate(features.unsqueeze(0), scale_factor=scale_factor, mode='linear').squeeze(0)
                fps = self.default_fps
                features, segments, labels = truncate_feats(
                    features, segments, labels,
                    max_seq_len=4096,
                    fps=fps,
                    base_frames=base_frames,
                    stride=stride,
                    crop_ratio=[0.9, 1.0],
                    trunc_thr=0.3,
                    max_num_trials=200,
                    has_action=True,
                    no_trunc=False,
                )

        elif self.mode == 'train':
            features, segments, labels = truncate_feats(
                features, segments, labels,
                max_seq_len=self.max_seq_len,
                fps=fps,
                base_frames=base_frames,
                stride=stride,
                crop_ratio=[0.9, 1.0],
                trunc_thr=0.3,
                max_num_trials=200,
                has_action=True,
                no_trunc=False,
            )

        feature_duration = features.size(1) * stride / fps

        if self.normalize:
            segments = segments / feature_duration

        if self.mode == 'train' and self.noise_scaler > 0:
            noise_scale = torch.rand_like(features) * self.noise_scaler + 1
            features = features * noise_scale

        if self.mode == 'train' and self.noise_scale > 0:
            features = features + self.noise_scale * torch.randn_like(features)

        info = {
            'video_name': video_name,
            'segments': segments,
            'labels': labels,
            'video_duration': torch.tensor(video_duration),
            'feature_duration': torch.tensor(feature_duration),
            'fps': torch.tensor(fps),
            'base_frames': torch.tensor(base_frames),
            'offset': torch.tensor(offset),
            'stride': torch.tensor(stride),
        }
        return features, info

def build_tad_dataset(subset, mode, ignored_videos, args):
    return TADDataset(
        args.feature_folder,
        args.gt_path,
        subset=subset,
        mode=mode,
        base_frame=args.base_frame,
        stride=args.stride,
        ignored_videos=ignored_videos,
        max_seq_len=args.max_seq_len,
        resize=args.resize,
        normalize=args.normalize,
        downsample_rate=args.downsample_rate,
        name_format=args.name_format,
        noise_scale=args.noise_scale,
        noise_scaler=args.noise_scaler,
        seg_noise_scale=args.seg_noise_scale,
        binary=args.binary,
        mem_cache=args.mem_cache,
        default_fps=args.default_fps,
    )