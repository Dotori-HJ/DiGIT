# TadTR: End-to-end Temporal Action Detection with Transformer

import concurrent.futures
import json
import torch
import logging
import os
import copy
import os.path as osp

# import ipdb as pdb
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .action_evaluator.eval_detection import compute_average_precision_detection
from util.misc import all_gather
from util.segment_ops import soft_nms, temporal_nms
from util.nms import apply_nms, apply_softnms

# from util.proposal_utils import soft_nms
from .data_util import get_dataset_dict, get_classes


def segment_tiou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
    return tiou

def average_recall_vs_nr_proposals(proposals,
                                   ground_truth,
                                   tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    """Computes the average recall given an average number of proposals per
    video.

    Parameters
    ----------
    proposals : DataFrame
        pandas table with the resulting proposals. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame,
                                'score': (float) Proposal confidence}
    ground_truth : DataFrame
        pandas table with annotations of the dataset. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame}
    tiou_thresholds : 1darray, optional
        array with tiou threholds.

    Outputs
    -------
    average_recall : 1darray
        recall averaged over a list of tiou threshold.
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    # Get list of videos.
    video_lst = proposals['video-id'].unique()

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:

        # Get proposals for this video.
        prop_idx = proposals['video-id'] == videoid
        this_video_proposals = proposals[prop_idx][['t-start', 't-end'
                                                    ]].values.astype(np.float64)

        # Sort proposals by score.
        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        # Get ground-truth instances associated to this video.
        gt_idx = ground_truth['video-id'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['t-start',
                                                        't-end']].values

        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    pcn_lst = np.arange(1, 201) / 200.0
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]

            for j, pcn in enumerate(pcn_lst):
                # Get number of proposals as a percentage of total retrieved.
                nr_proposals = int(score.shape[1] * pcn)
                # Find proposals that satisfies minimum tiou threhold.
                matches[i, j] = ((score[:, :nr_proposals] >= tiou).sum(axis=1)
                                 > 0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_lst * (float(proposals.shape[0]) /
                                     video_lst.shape[0])

    return recall, proposals_per_video

def eval_ap(iou, cls, gt, predition):
    ap = compute_average_precision_detection(gt, predition, iou)
    sys.stdout.flush()
    return cls, ap


# def apply_nms(dets_arr, nms_thr=0.4, use_soft_nms=False):
#     # the last column are class ids
#     unique_classes = np.unique(dets_arr[:, 3])
#     output_dets = []
#     for cls in unique_classes:
#         this_cls_dets = dets_arr[dets_arr[:, 3] == cls]
#         # this_cls_dets_kept = temporal_nms(this_cls_dets, nms_thr)
#         if not use_soft_nms:
#             this_cls_dets_kept = temporal_nms(this_cls_dets, nms_thr)
#         else:
#             this_cls_dets_kept = soft_nms(this_cls_dets, 0.7, 0, 0, 100)
#         output_dets.append(this_cls_dets_kept)
#     output_dets = np.concatenate(output_dets, axis=0)
#     sort_idx = output_dets[:, 2].argsort()[::-1]
#     output_dets = output_dets[sort_idx, :]
#     return output_dets

def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate / very short annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event['label']
        if (e - s) >= tol:
            valid = True
        else:
            valid = False
        for p_event in valid_events:
            if ((abs(s-p_event['segment'][0]) <= tol)
                and (abs(e-p_event['segment'][1]) <= tol)
                and (l == p_event['label'])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events

class TADEvaluator(object):
    def __init__(self, anno_path, subset, ignored_videos, extra_cls_path=None,
                 nms_mode=['raw'], iou_range=[0.5], display_metric_indices=[0],
                 nms_thr=0.5, nms_sigma=0.75, voting_thresh=-1, min_score=0.001, nms_multi_class=True, topk=200, num_workers=None, assign_cls_labels=False):
        '''dataset_name:  thumos14, activitynet or hacs
        subset: val or test
        video_dict: the dataset dict created in video_dataset.py
        iou_range: [0.3:0.7:0.1] for thumos14; [0.5:0.95:0.05] for anet and hacs.
        '''

        self.anno_path = anno_path
        self.subset = subset
        self.ignored_videos = ignored_videos
        self.extra_cls_path = extra_cls_path
        self.nms_mode = nms_mode
        self.iou_range = iou_range
        self.display_metric_indices = display_metric_indices
        self.nms_thr = nms_thr
        self.nms_sigma = nms_sigma
        self.voting_thresh = voting_thresh
        self.min_score = min_score
        self.nms_multi_class = nms_multi_class
        self.topk = topk
        self.assign_cls_labels = assign_cls_labels

        # if dataset_name == 'thumos14':
        #     subset_mapping = {'train': 'val', 'val': 'test'}
        #     anno_path = 'data/thumos14/th14_annotations_with_fps_duration.json'
        #     # follow SSN/PGCN/AFSD/MUSES to remove three falsely annotated videos
        #     self.ignored_videos = ['video_test_0000270', 'video_test_0001292', 'video_test_0001496']
        # elif dataset_name == 'activitynet':
        #     subset_mapping = {'train': 'training', 'val': 'validation'}
        #     # anno_file = 'data/activitynet/annotations/anet1.3_tsp_filtered.json'
        #     anno_path = 'data/activitynet/activity_net.v1-3.min.json'
        #     self.ignored_videos = []
        # elif dataset_name == 'hacs':
        #     subset_mapping = {'train': 'training', 'val': 'validation'}
        #     # anno_file = 'data/activitynet/annotations/anet1.3_tsp_filtered.json'
        #     anno_path = 'data/hacs/annotations/HACS_segments_v1.1.1_af.json'
        #     self.ignored_videos = []
        # elif dataset_name == 'epic_kitchens_noun':
        #     subset_mapping = {'train': 'training', 'val': 'validation'}
        #     anno_path = 'data/epic_kitchens/annotations/epic_kitchens_100_noun.json'
        #     self.ignored_videos = []
        # elif dataset_name == 'epic_kitchens_verb':
        #     subset_mapping = {'train': 'training', 'val': 'validation'}
        #     anno_path = 'data/epic_kitchens/annotations/epic_kitchens_100_verb.json'
        #     self.ignored_videos = []
        # else:
        #     raise NotImplementedError

        if extra_cls_path is not None:
            with open(extra_cls_path, 'rt') as f:
                self.cls_scores = json.load(f)['results']
        else:
            self.cls_scores = None
        # if dataset_name == 'activitynet':
        #     with open("data/activitynet/cuhk_val_simp_share.json", 'rt') as f:
        #         self.cls_scores = json.load(f)['results']
        # elif dataset_name == 'hacs':
        #     with open("data/hacs/annotations/tca_uniformerv2_large_only_global_anet_16x10x3.json", 'rt') as f:
        #         self.cls_scores = json.load(f)['results']

        with open(anno_path, 'rt') as f:
            self.gt = json.load(f)['database']

        self.video_names = []
        for video_name, video_info in self.gt.items():
            # Filtering out 'Ambiguous' annotations
            # video_info['annotations'] = remove_duplicate_annotations(video_info['annotations'])
            # video_info['annotations'] = [
            #     anno for anno in video_info['annotations']
            #     if anno['label'] != 'Ambiguous' and anno['segment'][0] < float(video_info['duration'])
            # ]
            if video_info['subset'] == subset and not video_name in ignored_videos:
                # Appending video name to the list after removing ambiguous annotations
                self.video_names.append(video_name)

        self.classes = get_classes(self.gt)
        self.num_classes = len(self.classes)
        self.classname_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_classname = {idx: cls for idx, cls in enumerate(self.classes)}

        # if 'CliffDiving' in self.classes:
        #     self.thumos14 = True
        #     self.cliff_idx = self.classes.index('CliffDiving')
        #     self.diving_idx = self.classes.index('Diving')
        # else:
        self.thumos14 = False

        all_gt = []
        self.durations = {}
        for video_name in self.video_names:
            annotations = self.gt[video_name]['annotations']
            all_gt += [
                (video_name, self.classname_to_idx[anno['label']], anno['segment'][0], anno['segment'][1])
                for anno in annotations
            ]
            self.durations[video_name] = self.gt[video_name]['duration']

        all_gt = pd.DataFrame(all_gt, columns=["video-id", "cls", "t-start", "t-end"])
        self.video_ids = all_gt['video-id'].unique().tolist()
        print('{} ground truth instances from {} videos'.format(len(all_gt), len(self.video_ids)))

        # per class ground truth
        gt_by_cls = []
        for cls in range(self.num_classes):
            gt_by_cls.append(all_gt[all_gt.cls == cls].reset_index(drop=True).drop(columns='cls'))

        self.gt_by_cls = gt_by_cls
        self.all_pred = {k: [] for k in self.nms_mode}
        self.all_gt = all_gt
        self.num_workers = self.num_classes if num_workers is None else num_workers
        self.stats = {k: dict() for k in self.nms_mode}

    def update(self, pred):
        '''pred: a dict of predictions for each video. For each video, the predictions are in a dict with these fields: scores, labels, segments
        assign_cls_labels: manually assign class labels to the detections. This is necessary when the predictions are class-agnostic.
        '''
        pred_numpy = {
            k: {
            kk: vv.detach().cpu().numpy()
            if not vv.dtype == torch.bfloat16
            else vv.float().detach().cpu().numpy()
            for kk, vv in v.items()
        } for k,v in pred.items()}
        for k, v in pred_numpy.items():
            this_dets = [
                [v['segments'][i, 0],
                    v['segments'][i, 1],
                    v['scores'][i], v['labels'][i]]
                    for i in range(len(v['scores']))]
            video_id = k

            if video_id not in self.video_ids:
                continue
            this_dets = np.array(this_dets)   # start, end, score, label

            for nms_mode in self.nms_mode:
                dets = np.copy(this_dets)
                # if nms_mode == 'nms' and not (cfg.TEST_SLICE_OVERLAP > 0 and self.dataset_name == 'thumos14'):  # when cfg.TEST_SLICE_OVERLAP > 0, only do nms at summarization
                # if nms_mode == 'nms' and not (cfg.test_slice_overlap > 0 and self.dataset_name == 'thumos14'):  # when cfg.TEST_SLICE_OVERLAP > 0, only do nms at summarization
                    # dets = apply_nms(input_dets, nms_thr=self.nms_thr, use_soft_nms=self.dataset_name=='activitynet')
                #     # min_score = 0.001
                #     # dets = dets[dets[:, 2] > min_score]
                # else:
                # input_dets = apply_nms(input_dets, nms_thr=self.nms_thr, use_soft_nms=False)
                # dets = dets[dets[:, 2] > self.min_score]

                if dets.shape[0] == 0:
                    continue

                # dets = input_dets
                if nms_mode == 'nms':
                    dets = apply_nms(dets, nms_thr=self.nms_thr, min_score=self.min_score, topk=self.topk, voting_thresh=self.voting_thresh, multi_class=self.nms_multi_class)
                if nms_mode == 'soft_nms':
                    dets = apply_softnms(dets, nms_thr=0.1, min_score=self.min_score, topk=self.topk, sigma=self.nms_sigma, voting_thresh=self.voting_thresh, multi_class=self.nms_multi_class)

                # only keep top 300 detections per video
                # if self.topk > 0:
                #     dets = dets[:self.topk, :]
                dets = dets[dets[:, 2] > self.min_score]
                sort_idx = dets[:, 2].argsort()[::-1]
                dets = dets[sort_idx, :]
                if self.thumos14:
                    if (dets[:, -1] == self.cliff_idx).any():
                        # duplicate cliff diving
                        cliff_dets = copy.deepcopy(dets[dets[:, -1] == self.cliff_idx])
                        cliff_dets[:, -1] = self.diving_idx
                        dets = np.concatenate([dets, cliff_dets])

                if self.topk > 0:
                    dets = dets[:self.topk, :]

                # On ActivityNet, follow the tradition to use external video label
                if self.cls_scores is not None:
                    if self.assign_cls_labels:

                        cls_scores = np.asarray(self.cls_scores[video_id])
                        # if self.dataset_name == 'hacs':
                            # import torch
                            # cls_scores = torch.sigmoid(torch.from_numpy(cls_scores)).numpy()
                        topk = (cls_scores > 0.05).sum().item()
                        # topk = (cls_scores > 0.01).sum().item()
                        # topk = 2
                        topk_cls_idx = np.argsort(cls_scores)[::-1][:topk]
                        topk_cls_score = cls_scores[topk_cls_idx]

                        # duplicate all segment and assign the topk labels
                        # K x 1 @ 1 N -> K x N -> KN
                        # multiply the scores
                        new_pred_score = np.sqrt(topk_cls_score[:, None] @ dets[:, 2][None, :]).flatten()[:, None]
                        new_pred_segment = np.tile(dets[:, :2], (topk, 1))
                        new_pred_label = np.tile(topk_cls_idx[:, None], (1, len(dets))).flatten()[:, None]
                        dets = np.concatenate((new_pred_segment, new_pred_score, new_pred_label), axis=-1)
                    else:
                        try:
                            cls_scores = np.asarray(self.cls_scores[video_id])
                            # topk = 2
                            topk = (cls_scores > 0.05).sum().item()
                            topk_cls_idx = np.argsort(cls_scores)[::-1][:topk]
                            dets = np.concatenate([dets[dets[:, 3] == idx] for idx in topk_cls_idx])
                        except:
                            pass


                self.all_pred[nms_mode] += [[video_id] + det for det in dets.tolist()]

    def nms_whole_dataset(self):
        # video_ids = list(set([v['src_vid_name'] for k, v in self.video_dict.items()]))
        video_ids = self.all_pred['nms']['video-id'].unique()
        # video_ids = self.video_ids
        all_pred = []
        for vid in video_ids:
            this_dets = self.all_pred['nms'][self.all_pred['nms']['video-id'] == vid][['t-start', 't-end', 'score', 'cls']].values
            # num_queries = len(this_dets)

            this_dets = this_dets[this_dets[..., 0] < self.durations[vid]]
            this_dets = this_dets[this_dets[..., 1] > 0]
            this_dets = np.clip(this_dets, a_min=0, a_max=self.durations[vid])
            # this_dets = this_dets[this_dets[..., 1] - this_dets[..., 0] > 0.1]
            # min_score = 0.001
            # this_dets = this_dets[this_dets[..., 2] > min_score]

            this_dets = apply_nms(this_dets, nms_thr=self.nms_thr)
            if self.topk > 0:
                this_dets = this_dets[:self.topk, ...]
            # this_dets = this_dets[:int(self.durations[vid] * 1.5), ...]
            # this_dets = this_dets[:num_queries // 5, ...]
            this_dets = [[vid] + x.tolist() for x in this_dets]
            all_pred += this_dets
        self.all_pred['nms'] = pd.DataFrame(all_pred, columns=["video-id", "t-start", "t-end", "score", "cls"])

    def accumulate(self):
        '''accumulate detections in all videos'''
        for nms_mode in self.nms_mode:
            self.all_pred[nms_mode] = pd.DataFrame(self.all_pred[nms_mode], columns=["video-id", "t-start", "t-end", "score", "cls"])

        self.pred_by_cls = {}
        for nms_mode in self.nms_mode:
            # if self.dataset_name == 'thumos14' and nms_mode == 'raw' and test_slice_overlap > 0:
            #     self.cross_window_fusion()
            # # if you really want to use NMS
            # if nms_mode == 'nms':
            #     self.nms_whole_dataset()

            self.pred_by_cls[nms_mode] = [self.all_pred[nms_mode][self.all_pred[nms_mode].cls == cls].reset_index(drop=True).drop(columns='cls') for cls in range(self.num_classes)]

    def import_prediction(self):
        pass

    def format_arr(self, arr, format='{:.2f}'):
        line = ' '.join([format.format(x) for x in arr])
        return line

    def synchronize_between_processes(self):
        self.all_pred = merge_distributed(self.all_pred)

    def summarize(self):
        '''Compute mAP and collect stats'''
        for nms_mode in self.nms_mode:
            print(
                'mode={} {} predictions from {} videos'.format(
                    nms_mode,
                    len(self.all_pred[nms_mode]),
                    len(self.all_pred[nms_mode]['video-id'].unique()))
            )

        header = ' '.join('%.2f' % self.iou_range[i] for i in self.display_metric_indices) + ' avg'  # 0 5 9
        lines = []
        for nms_mode in self.nms_mode:
            per_iou_ap = self.compute_map(nms_mode)
            line = ' '.join(['%.2f' % (100*per_iou_ap[i]) for i in self.display_metric_indices]) + ' %.2f' % (100*per_iou_ap.mean()) + ' {}'.format(nms_mode)
            lines.append(line)
        msg = header
        for l in lines:
            msg += '\n' + l
        print('\n' + msg)

        # ar_metrics = {}
        # for nms_mode in self.nms_mode:
        #     # Get predictions for this mode
        #     pred_df = self.all_pred[nms_mode]

        #     # Assume ground_truth is in the same format as pred_df
        #     ground_truth = self.all_gt

        #     # Calculate AR@50 and AR@100
        #     average_recall, average_nr_proposals = average_recall_vs_nr_proposals(
        #         pred_df, ground_truth)

        #     # Interpolate to find AR@50 and AR@100
        #     f = interp1d(average_nr_proposals, average_recall, fill_value='extrapolate')
        #     ar_metrics[nms_mode] = {
        #         'AR@50': f(50).item(),
        #         'AR@100': f(100).item(),
        #         'AR@200': f(200).item(),
        #         'AR@500': f(500).item(),
        #     }

        # for nms_mode, metrics in ar_metrics.items():
        #     print(f"{nms_mode} - AR@50: {metrics['AR@50']:.4f}, AR@100: {metrics['AR@100']:.4f}, AR@200: {metrics['AR@200']:.4f}, AR@500: {metrics['AR@500']:.4f}")


        for nms_mode in self.nms_mode:
            ap50_idx = self.iou_range.index(0.5)
            self.stats[nms_mode]['AP50'] = self.stats[nms_mode]['per_iou_ap'][ap50_idx]
        self.stats_summary = msg

    def compute_map(self, nms_mode):
        '''Compute mean average precision'''
        start_time = time.time()

        gt_by_cls, pred_by_cls = self.gt_by_cls, self.pred_by_cls[nms_mode]

        valid_gt_classes = []
        class_mapper = {}
        for cls in range(len(pred_by_cls)):
            if not len(gt_by_cls[cls]) == 0:
                class_mapper[cls] = len(valid_gt_classes)
                valid_gt_classes.append(cls)

        iou_range = self.iou_range
        # num_classes = self.num_classes
        num_classes = len(valid_gt_classes)
        ap_values = np.zeros((num_classes, len(iou_range)))

        # with concurrent.futures.ProcessPoolExecutor(min(self.num_workers, 8)) as p:
        with concurrent.futures.ProcessPoolExecutor(min(self.num_workers, os.cpu_count())) as p:
        # with concurrent.futures.ProcessPoolExecutor(self.num_workers) as p:
            futures = []
            # for cls in range(len(pred_by_cls)):
            for i, cls in enumerate(valid_gt_classes):
                if len(gt_by_cls[cls]) == 0:
                    print('no gt for class {}'.format(self.classes[cls]))
                if len(pred_by_cls[cls]) == 0:
                    print('no prediction for class {}'.format(self.classes[cls]))
                futures.append(p.submit(eval_ap, iou_range, cls, gt_by_cls[cls], pred_by_cls[cls]))
            for f in concurrent.futures.as_completed(futures):
                x = f.result()
                ap_values[class_mapper[x[0]], :] = x[1]

        per_iou_ap = ap_values.mean(axis=0)
        per_cls_ap = ap_values.mean(axis=1)
        mAP = per_cls_ap.mean()

        self.stats[nms_mode]['mAP'] = mAP
        self.stats[nms_mode]['ap_values'] = ap_values.tolist()
        self.stats[nms_mode]['per_iou_ap'] = per_iou_ap.tolist()
        self.stats[nms_mode]['per_cls_ap'] = per_cls_ap.tolist()
        return per_iou_ap

    def dump_to_json(self, dets, save_path):
        result_dict = {}
        videos = dets['video-id'].unique()
        for video in videos:
            this_detections = dets[dets['video-id'] == video]
            det_list = []
            for idx, row in this_detections.iterrows():
                det_list.append(
                    {'segment': [float(row['t-start']), float(row['t-end'])], 'label': self.classes[int(row['cls'])], 'score': float(row['score'])}
                )

            video_id = video[2:] if video.startswith('v_') else video
            result_dict[video_id] = det_list

        # the standard detection format for ActivityNet
        output_dict={
            "version": "VERSION 1.3",
            "results": result_dict,
            "external_data":{}}
        if save_path:
            # dirname = osp.dirname(save_path)
            # if not osp.exists(dirname):
            #     os.makedirs(dirname)
            with open(save_path, 'w') as f:
                json.dump(output_dict, f)
        # return output_dict

    def dump_detection(self, save_path=None):
        for nms_mode in self.nms_mode:
            print(
                'dump detection result in JSON format to {}'.format(save_path.format(nms_mode)))
            self.dump_to_json(self.all_pred[nms_mode], save_path.format(nms_mode))


def merge_distributed(all_pred):
    '''gather outputs from different nodes at distributed mode'''
    all_pred_gathered = all_gather(all_pred)

    merged_all_pred = {k: [] for k in all_pred}
    for p in all_pred_gathered:
        for k in p:
            merged_all_pred[k] += p[k]

    return merged_all_pred


if __name__ == '__main__':
    pass


