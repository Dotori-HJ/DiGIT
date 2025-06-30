# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for segment manipulation and IoU.
"""
import warnings
import torch.nn.functional as F
import numpy as np
import torch

# from torchvision.ops.boxes import box_area

def calculate_center_times_and_coverage(feature_length, fps=30, window_size=16, stride=4):
    # Create feature indices: [0, 1, 2, ..., total_features - 1]
    feature_indices = torch.arange(0, feature_length)

    # Calculate the center frame index for each feature
    center_frame_indices = feature_indices * stride + window_size // 2

    # Convert the center frame indices to time in seconds
    center_times = center_frame_indices.float() / fps

    # Calculate the total time coverage: window_size / fps
    pad = window_size / (2 * fps)
    total_time_coverage = center_times[-1] - center_times[0] +  2 * pad

    return center_times, total_time_coverage

def get_feature_grid(feature_length, fps=30, window_size=16, stride=4):
    # Create feature indices: [0, 1, 2, ..., total_features - 1]
    feature_indices = torch.arange(0, feature_length)

    # Calculate the center frame index for each feature
    center_frame_indices = feature_indices * stride + window_size // 2

    # Convert the center frame indices to time in seconds
    feature_grid = center_frame_indices.float() / fps

    return feature_grid

def get_feature_time_coverage(feature_length, fps=30, window_size=16, stride=4):
    return feature_length * stride / fps
    # return ((feature_length - 1) * stride + window_size * 0.25) / fps


def get_time_coverage(feature_length, fps=30, window_size=16, stride=4):
    return ((feature_length - 1) * stride + window_size) / fps
    # return ((feature_length - 1) * stride + (window_size - stride) * 2) / fps


def compute_segment_targets(anchors, targets):
    """
    Compute relative segment encoding targets given anchors and ground truth segments.
    Params:
        anchors: Tensor of shape [num_segments, 2], each row is (act, aw) representing an anchor.
        targets: Tensor of shape [num_segments, 2], each row is (t_start, t_end) representing a ground truth segment.
    Returns:
        targets_dt_dw: Tensor of shape [num_segments, 2], each row is (dt, dw) representing the relative change.
    """
    act, aw = anchors.unbind(-1)
    t_start, t_end = targets.unbind(-1)

    ct_gt = (t_start + t_end) / 2
    w_gt = t_end - t_start

    dt = ct_gt - act
    dw = torch.log(w_gt / aw)

    return torch.stack([dt, dw], dim=-1)

def compute_pairwise_segment_targets(anchors, targets):
    """
    Compute pairwise regression targets given predictions, anchors, and ground truth segments.
    Params:
        predicted_segments: Tensor of shape [num_queries, 2] representing predicted segments.
        anchors: Tensor of shape [num_queries, 2], each row is (act, aw) representing an anchor.
        targets: Tensor of shape [num_queries, 2], each row is (t_start, t_end) representing a ground truth segment.
    Returns:
        pairwise_targets_dt_dw: Tensor of shape [num_queries, num_queries, 2], where [i, j] gives (dt, dw) for
                                the i-th prediction with respect to the j-th target.
    """

    # Reshape anchors and targets for broadcasting
    targets = targets[None, :, :]  # Shape [1, num_queries, 2]

    # Compute regression targets based on difference between anchors and targets
    dt = (targets[..., 0] - anchors[..., 0]) / anchors[..., 1]
    dw = torch.log(targets[..., 1] / anchors[..., 1])

    pairwise_targets_dt_dw = torch.stack([dt, dw], dim=-1)  # Shape [num_queries, num_queries, 2]

    return pairwise_targets_dt_dw

def segment_dtdwctw_to_cw(x, a=None):
    '''corresponds to box_cxcywh_to_xyxy in detr
    Params:
        x: segments in (center, width) format, shape=(*, 2)
    Returns:
        segments in (t_start, t_end) format, shape=(*, 2)
    '''
    if a is None:
        dt, dw, act, aw = x.unbind(-1)
    else:
        dt, dw = x.unbind(-1)
        act, aw = a.unbind(-1)

    ct = act + dt * aw
    # ct = act + dt
    # w = F.softplus(dw)
    w = aw * dw.exp()
    # w = aw * F.softplus(dw)
    # w = torch.where(w > 0, w, torch.full_like(w, fill_value=0.01))
    b = torch.stack([ct, w], dim=-1)
    return b

def segment_dtdwctw_to_t1t2(x, a=None):
    '''corresponds to box_cxcywh_to_xyxy in detr
    Params:
        x: segments in (center, width) format, shape=(*, 2)
    Returns:
        segments in (t_start, t_end) format, shape=(*, 2)
    '''
    if a is None:
        dt, dw, act, aw = x.unbind(-1)
    else:
        dt, dw = x.unbind(-1)
        act, aw = a.unbind(-1)

    ct = act + dt * aw
    # ct = act + dt
    # w = F.softplus(dw)
    w = aw * dw.exp()
    # w = aw * F.softplus(dw)
    # w = torch.where(w > 0, w, torch.full_like(w, fill_value=0.01))
    b = torch.stack([ct - w * 0.5, ct + w * 0.5], dim=-1)
    return b

@torch.jit.script
def inverse_log_and_stack(x):
    return torch.stack((x[..., 0], x[..., 1].exp()), dim=-1)

# def inverse_softplus(x, beta=1.0):
#     return (x * beta).expm1().clamp_min(1e-6).log() / beta

def inverse_softplus(x):
    return x.expm1().clamp_min(1e-6).log()

def segment_cw_to_t1t2(x):
    '''corresponds to box_cxcywh_to_xyxy in detr
    Params:
        x: segments in (center, width) format, shape=(*, 2)
    Returns:
        segments in (t_start, t_end) format, shape=(*, 2)
    '''
    if not isinstance(x, np.ndarray):
        x_c, w = x.unbind(-1)
        b = [(x_c - 0.5 * w), (x_c + 0.5 * w)]
        b = torch.stack(b, dim=-1)
        return b
    else:
        x_c, w = x[..., 0], x[..., 1]
        b = [(x_c - 0.5 * w)[..., None], (x_c + 0.5 * w)[..., None]]
        return np.concatenate(b, axis=-1)


def segment_clr_to_t1t2(x):
    '''corresponds to box_cxcywh_to_xyxy in detr
    Params:
        x: segments in (center, width) format, shape=(*, 2)
    Returns:
        segments in (t_start, t_end) format, shape=(*, 2)
    '''
    if not isinstance(x, np.ndarray):
        x_c, l, r = x.unbind(-1)
        # w = torch.where(w > 0, w, torch.full_like(w, fill_value=0.01))
        b = [(x_c - l), (x_c + r)]
        b = torch.stack(b, dim=-1)
        # mask = b[:, 1] == torch.tensor(float('inf'))
        # b[mask].fill_(0.01)
        return b
    else:
        x_c, l, r = x[..., 0], x[..., 1], x[..., 2]
        b = [(x_c - l)[..., None], (x_c + r)[..., None]]
        return np.concatenate(b, axis=-1)

def segment_t1t2_to_cw(x):
    '''corresponds to box_xyxy_to_cxcywh in detr
    Params:
        x: segments in (t_start, t_end) format, shape=(*, 2)
    Returns:
        segments in (center, width) format, shape=(*, 2)
    '''
    if not isinstance(x, np.ndarray):
        x1, x2 = x.unbind(-1)
        b = [(x1 + x2) / 2, (x2 - x1)]
        return torch.stack(b, dim=-1)
    else:
        x1, x2 = x[..., 0], x[..., 1]
        b = [((x1 + x2) / 2)[..., None], (x2 - x1)[..., None]]
        return np.concatenate(b, axis=-1)

def segment_clr_to_t1t2(x):
    '''corresponds to box_cxcywh_to_xyxy in detr
    Params:
        x: segments in (center, width) format, shape=(*, 2)
    Returns:
        segments in (t_start, t_end) format, shape=(*, 2)
    '''
    if not isinstance(x, np.ndarray):
        x_c, l, r = x.unbind(-1)
        b = [(x_c - l), (x_c + r)]
        return torch.stack(b, dim=-1)
    else:
        x_c, l, r = x[..., 0], x[..., 1], x[..., 2]
        b = [(x_c - l)[..., None], (x_c + r)[..., None]]
        return np.concatenate(b, axis=-1)

@torch.jit.script
def segment_length(segments):
    return (segments[:, 1]-segments[:, 0]).clamp(min=0)


# modified from torchvision to also return the union
@torch.jit.script
def segment_iou_and_union(segments1, segments2):
    area1 = segment_length(segments1)
    area2 = segment_length(segments2)

    l = torch.max(segments1[:, None, 0], segments2[:, 0])  # N,M
    r = torch.min(segments1[:, None, 1], segments2[:, 1])  # N,M
    inter = (r - l).clamp(min=0)  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

@torch.jit.script
def segment_iou(segments1, segments2):
    """
    Temporal IoU between

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(segments1)
    and M = len(segments2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (segments1[:, 1] >= segments1[:, 0]).all(), f'{segments1.size()}_{(segments1[:, 1] >= segments1[:, 0]).sum()}_{segments1[~(segments1[:, 1] >= segments1[:, 0])]}'

    area1 = segment_length(segments1)
    area2 = segment_length(segments2)

    l = torch.max(segments1[:, None, 0], segments2[:, 0])  # N,M
    r = torch.min(segments1[:, None, 1], segments2[:, 1])  # N,M
    inter = (r - l).clamp(min=0)  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union

    return iou

@torch.jit.script
def segment_diou(src_segments: torch.Tensor, target_segments: torch.Tensor) -> torch.Tensor:
    """
    Calculate the direct DIOU between predicted and target segments.

    Args:
    - predicted_segments: Tensor of shape (N, 2), where N is the number of segments.
    - target_segments: Tensor of shape (N, 2), where N is the number of segments.

    Returns:
    - diou: Tensor of shape (N,), where each value is the DIOU between the corresponding predicted and target segment.
    """
    # Intersection
    l = torch.max(src_segments[:, 0], target_segments[:, 0])
    r = torch.min(src_segments[:, 1], target_segments[:, 1])
    inter = (r - l).clamp(min=0)

    # Union
    area_pred = src_segments[:, 1] - src_segments[:, 0]
    area_target = target_segments[:, 1] - target_segments[:, 0]
    union = area_pred + area_target - inter

    # IoU
    iou = inter / union

    # Center distance
    center_pred = (src_segments[:, 0] + src_segments[:, 1]) / 2
    center_target = (target_segments[:, 0] + target_segments[:, 1]) / 2
    center_dist = torch.abs(center_pred - center_target)

    # Max possible segment length
    max_length = torch.max(src_segments[:, 1], target_segments[:, 1]) - torch.min(src_segments[:, 0], target_segments[:, 0])

    # Normalize the center distance by the max_length
    normalized_dist = center_dist / max_length

    # DIOU
    diou = iou - normalized_dist**2

    return diou


@torch.jit.script
def diou_loss(src_segments: torch.Tensor, target_segments: torch.Tensor) -> torch.Tensor:
    diou_values = segment_diou(src_segments, target_segments)
    loss_values = 1.0 - diou_values
    return loss_values

def center_width_loss(src_segments: torch.Tensor, target_segments: torch.Tensor, beta: float = 1.0, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Calculate the smooth L1 log ratio width loss between predicted and target segments.
    """
    center_pred = src_segments[..., :1]
    width_pred = src_segments[..., 1:]
    center_target = target_segments[..., :1]
    width_target = target_segments[..., 1:].log()
    
    # center_t = (center_target - center_pred) / width_pred.exp()
    # loss_center = F.smooth_l1_loss(center_t, torch.zeros_like(center_t), reduction='none')

    # width_ratio = width_pred / width_target
    loss_width = F.smooth_l1_loss(width_pred, width_target, reduction='none')

    return loss_width
    # return loss_width

def pairwise_log_ratio_width_loss(src_segments: torch.Tensor, target_segments: torch.Tensor, beta: float = 1.0, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Calculate the pairwise smooth L1 log ratio width loss between predicted and target segments.
    Returns a tensor of shape [N, M] where N is the number of source segments and M is the number of target segments.
    """
    # center_pred = src_segments[:, None, 0]
    width_pred = src_segments[:, None, 1]
    # center_target = target_segments[None, :, 0]
    width_target = target_segments[None, :, 1].log()

    # loss = torch.abs(center_target - center_pred) 
    # center_t = (center_target - center_pred) / width_pred.exp()
    # loss = F.smooth_l1_loss(center_t, torch.zeros_like(center_t), reduction='none')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # loss = F.smooth_l1_loss(width_pred, width_target, reduction='none') + loss
        loss = F.smooth_l1_loss(width_pred, width_target, reduction='none')

    return loss


# def log_ratio_width_loss(src_segments: torch.Tensor, target_segments: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
#     """
#     Calculate the log ratio width loss between predicted and target segments.
#     """
#     width_pred = src_segments[..., 1]
#     width_target = target_segments[..., 1]

#     log_width_ratio = torch.log(width_pred / width_target)

#     return log_width_ratio ** 2

@torch.jit.script
def pairwise_segment_diou(src_segments: torch.Tensor, target_segments: torch.Tensor) -> torch.Tensor:
    """
    Calculate the pairwise DIOU between predicted and target segments.

    Args:
    - src_segments: Tensor of shape (N, 2), where N is the number of predicted segments.
    - target_segments: Tensor of shape (M, 2), where M is the number of target segments.

    Returns:
    - diou: Tensor of shape (N, M), where each value is the DIOU between the corresponding predicted and target segment.
    """
    # Intersection
    l = torch.max(src_segments[:, None, 0], target_segments[:, 0])
    r = torch.min(src_segments[:, None, 1], target_segments[:, 1])
    inter = (r - l).clamp(min=0)

    # Union
    area_pred = (src_segments[:, None, 1] - src_segments[:, None, 0])
    area_target = (target_segments[:, 1] - target_segments[:, 0])
    union = area_pred + area_target - inter

    # IoU
    iou = inter / union

    # Center distance
    center_pred = (src_segments[:, None, 0] + src_segments[:, None, 1]) / 2
    center_target = (target_segments[:, 0] + target_segments[:, 1]) / 2
    center_dist = torch.abs(center_pred - center_target)

    # Max possible segment length
    max_length = torch.max(src_segments[:, None, 1], target_segments[:, 1]) - torch.min(src_segments[:, None, 0], target_segments[:, 0])

    # Normalize the center distance by the max_length
    normalized_dist = center_dist / max_length

    # DIOU
    diou = iou - normalized_dist**2

    return diou


def temporal_iou_numpy(proposal_min, proposal_max, gt_min, gt_max):
    """Compute IoU score between a groundtruth instance and the proposals.

    Args:
        proposal_min (list[float]): List of temporal anchor min.
        proposal_max (list[float]): List of temporal anchor max.
        gt_min (float): Groundtruth temporal box min.
        gt_max (float): Groundtruth temporal box max.

    Returns:
        list[float]: List of iou scores.
    """
    len_anchors = proposal_max - proposal_min
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    union_len = len_anchors - inter_len + gt_max - gt_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def temporal_iou_numpy(proposal_min, proposal_max, gt_min, gt_max):
    """Compute IoP score between a groundtruth bbox and the proposals.

    Compute the IoP which is defined as the overlap ratio with
    groundtruth proportional to the duration of this proposal.

    Args:
        proposal_min (list[float]): List of temporal anchor min.
        proposal_max (list[float]): List of temporal anchor max.
        gt_min (float): Groundtruth temporal box min.
        gt_max (float): Groundtruth temporal box max.

    Returns:
        list[float]: List of intersection over anchor scores.
    """
    len_anchors = np.array(proposal_max - proposal_min)
    int_tmin = np.maximum(proposal_min, gt_min)
    int_tmax = np.minimum(proposal_max, gt_max)
    inter_len = np.maximum(int_tmax - int_tmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def soft_nms(proposals, alpha, low_threshold, high_threshold, top_k, min_score=0.001):
    """Soft NMS for temporal proposals.

    Args:
        proposals (np.ndarray): Proposals generated by network.
        alpha (float): Alpha value of Gaussian decaying function.
        low_threshold (float): Low threshold for soft nms.
        high_threshold (float): High threshold for soft nms.
        top_k (int): Top k values to be considered.

    Returns:
        np.ndarray: The updated proposals.
    """
    proposals = proposals[proposals[:, -1].argsort()[::-1]]
    tstart = list(proposals[:, 0])
    tend = list(proposals[:, 1])
    tscore = list(proposals[:, 2])
    tclasses = list(proposals[:, 3])
    rstart = []
    rend = []
    rscore = []
    rclasses =  []

    while len(tscore) > 0 and len(rscore) < top_k:
        max_index = np.argmax(tscore)
        max_width = tend[max_index] - tstart[max_index]
        iou_list = temporal_iou_numpy(tstart[max_index], tend[max_index],
                                      np.array(tstart), np.array(tend))
        iou_exp_list = np.exp(-np.square(iou_list) / alpha)

        for idx, _ in enumerate(tscore):
            if idx != max_index:
                current_iou = iou_list[idx]
                if current_iou > low_threshold + (high_threshold -
                                                  low_threshold) * max_width:
                    tscore[idx] = tscore[idx] * iou_exp_list[idx]

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        rclasses.append(tclasses[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
        tclasses.pop(max_index)

    rstart = np.array(rstart).reshape(-1)
    rend = np.array(rend).reshape(-1)
    rscore = np.array(rscore).reshape(-1)
    rclasses = np.array(rclasses).reshape(-1)
    indices = rscore > min_score
    new_proposals = np.stack((rstart[indices], rend[indices], rscore[indices], rclasses[indices]), axis=1)
    return new_proposals


def temporal_nms(segments, thresh):
    """
    One-dimensional non-maximal suppression
    :param segments: [[st, ed, score, ...], ...]
    :param thresh:
    :return:
    """
    t1 = segments[:, 0]
    t2 = segments[:, 1]
    scores = segments[:, 2]

    durations = t2 - t1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        IoU = intersection / \
            (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    return segments[keep, :]
