# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .tad_dataset import build_tad_dataset
# from .tad_dataset_e2e import build_video_dataset
from .action_eval import TADEvaluator

def get_dataset_info(args):
    if 'thumos14' in args.dataset_name:
        subset_mapping = {'train': 'val', 'val': 'test'}
        ignored_videos = ['video_test_0000270', 'video_test_0001292', 'video_test_0001496']
    else:
        subset_mapping = {'train': 'training', 'val': 'validation'}
        ignored_videos = []
    return subset_mapping, ignored_videos


def build_evaluator(subset, args):
    subset_mapping, ignored_videos = get_dataset_info(args)
    
    return TADEvaluator(
        args.gt_path, subset_mapping[subset], ignored_videos, args.extra_cls_path,
        args.nms_mode, args.iou_range, args.display_metric_indices,
        args.nms_thr, args.nms_sigma, args.voting_thresh, args.min_score, args.nms_multi_class, args.eval_topk, args.eval_workers, args.binary,
    )

def build_dataset(subset, mode, args):
    subset_mapping, ignored_videos = get_dataset_info(args)
    if args.datatype == 'feature':
        return build_tad_dataset(subset_mapping[subset], mode, ignored_videos, args)
    elif args.datatype == 'video':
        return build_video_dataset(subset_mapping[subset], mode, ignored_videos, args)
    assert False, 'Invalid dataset type'