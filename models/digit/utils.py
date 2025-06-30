# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from torch import nn, Tensor

import math
import torch.nn.functional as F
from torch import nn



def dice_loss(inputs, targets, num_segments, mask=None, smooth=1e-5):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    if mask is not None:
        m = ~mask
        inputs = inputs.masked_fill(m.unsqueeze(-1), 0)

    numerator = 2 * (inputs * targets).sum(dim=1)
    denominator = inputs.sum(dim=1) + targets.sum(dim=1)
    loss = (numerator + 1) / (denominator + 1)
    return loss.sum() / num_segments


def seg_voting(all_segs, all_scores, iou_threshold):
    """
        Improve localization results by incorporating all segs through voting,
        boosting the performance around iou_threshold without using NMS.
    """

    # compute overlap between all segs with each other
    num_segs = all_segs.shape[0]
    ex_all_segs1 = all_segs[:, None].expand(num_segs, num_segs, 2)
    ex_all_segs2 = all_segs[None, :].expand(num_segs, num_segs, 2)

    # compute intersection
    left = torch.maximum(ex_all_segs1[:, :, 0], ex_all_segs2[:, :, 0])
    right = torch.minimum(ex_all_segs1[:, :, 1], ex_all_segs2[:, :, 1])
    inter = (right - left).clamp(min=0)

    # lens of segments
    seg_lens1 = ex_all_segs1[:, :, 1] - ex_all_segs1[:, :, 0]
    seg_lens2 = ex_all_segs2[:, :, 1] - ex_all_segs2[:, :, 0]

    # iou
    iou = inter / (seg_lens1 + seg_lens2 - inter)

    # get neighbors (weights)
    weights = (iou >= iou_threshold).float() * all_scores[None, :] * iou
    weights /= torch.sum(weights, dim=1, keepdim=True)

    # refine all segments
    refined_segs = weights @ all_segs

    return refined_segs


def get_feature_grids(mask, fps, window_size, stride):
    B, T = mask.size()
    # Create feature indices: [0, 1, 2, ..., total_features - 1]
    feature_indices = torch.arange(0, T, dtype=torch.float32, device=mask.device)
    feature_indices.unsqueeze_(0)

    feature_indices = feature_indices.repeat(B, 1)

    # Calculate the center frame index for each feature
    center_frame_indices = feature_indices * stride[:, None] + window_size[:, None] // 2

    # last_frame_indcies = feature_indices[:, [-1]] * stride[:, None] + window_size[:, None]
    # print(last_frame_indcies[0], center_frame_indices.max().item(), T, fps[0], window_size[0], stride[0])
    # Convert the center frame indices to time in seconds

    feature_grid = center_frame_indices / fps[:, None]
    return feature_grid

    # return feature_grid.masked_fill_(mask, 0)



class RandomBoxPerturber():
    def __init__(self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2) -> None:
        self.noise_scale = torch.Tensor([x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale])

    def __call__(self, refanchors: Tensor) -> Tensor:
        nq, bs, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = torch.rand_like(refanchors)
        noise_scale = self.noise_scale.to(device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.clamp_(0, 1)


@torch.jit.script
def smooth_l1_loss(input, target, beta: float = 1. / 9):
    """
    Smooth L1 Loss defined as:
        |x| < beta => 0.5 * x^2 / beta
        otherwise => |x| - 0.5 * beta
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    return loss



def sigmoid_focal_loss(inputs, targets, mask, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if mask is not None:
        valid = ~mask[..., None]
        loss = loss * valid.float()

        num_queries = valid.sum(1).float()
        normalizer = num_queries.clamp(min=1.0)
        return ((loss.sum(1) / normalizer).sum() / num_boxes) * num_queries.mean()
    else:
        num_queries = inputs.size(1)
        return (loss.mean(1).sum() / num_boxes) * num_queries

    if mask is not None:
        loss = loss * ().float()
        mask
        normalizer = mask.sum(1).float().clamp(min=1.0)  # To avoid division by 0
        return ((loss.sum(1) / normalizer).sum() / num_boxes) * num_queries
    else:
        num_queries = inputs.size(1)

        return (loss.mean(1).sum() / num_boxes) * num_queries


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            # x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    if activation == "silu":
        return F.silu

    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos