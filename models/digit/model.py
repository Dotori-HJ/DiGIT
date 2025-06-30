# ------------------------------------------------------------------------
# DiGIT: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (httpsslow://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DiGIT model and criterion classes.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .matcher import build_matcher
from .utils import get_feature_grids, MLP
from .position_encoding import build_position_encoding
from util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from ..registry import MODULE_BUILD_FUNCS
from .modules import ConvBackbone

from util.segment_ops import (
    segment_cw_to_t1t2, segment_t1t2_to_cw, segment_iou, diou_loss, center_width_loss
)
from .modules import GatedConv

from .utils import sigmoid_focal_loss
from .transformer import build_deformable_transformer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm_type, dim, num_groups=None):
    if norm_type == 'gn':
        assert num_groups is not None, 'num_groups must be specified'
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == 'bn':
        return nn.BatchNorm1d(dim)
    else:
        raise NotImplementedError


class DiGIT(nn.Module):
    """ This is the DiGIT module that performs temporal action detection """

    def __init__(self, position_embedding, transformer, num_classes, num_queries, feature_dim, num_feature_levels,
                 num_sampling_levels, kernel_size=3, num_cls_head_layers=3, num_reg_head_layers=3,
                 aux_loss=True, with_segment_refine=False,
                 emb_norm_type='bn', emb_relu=False, share_class_embed=False, share_segment_embed=False,
                 fix_encoder_proposals=True, emb_dropout=0.0):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See deformable_transformer.py
            num_classes: number of action classes
            num_queries: number of action queries, ie detection slot. This is the maximal number of actions
                         DiGIT can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_segment_refine: iterative segment refinement
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.feature_dim = feature_dim
        self.num_sampling_levels = num_sampling_levels
        self.num_feature_levels = num_feature_levels
        self.emb_norm_type = emb_norm_type
        self.emb_relu = emb_relu
        self.fix_encoder_proposals = fix_encoder_proposals
        self.num_cls_head_layers = num_cls_head_layers
        self.num_reg_head_layers = num_reg_head_layers

        self.input_proj = ConvBackbone(
            # feature_dim, hidden_dim, kernel_size=kernel_size,
            feature_dim, hidden_dim, kernel_size=kernel_size,
            arch=(1, 0), num_feature_levels=1,
            with_ln=True, dropout=emb_dropout,
        )
        # self.input_gc = GatedConv(hidden_dim, kernel_size=7, num_levels=4, expansion_factor=1)
        self.position_embedding = position_embedding
        self.transformer.position_embedding = self.position_embedding
        self.aux_loss = aux_loss
        self.with_segment_refine = with_segment_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        if num_cls_head_layers > 1:
            self.class_embed = MLP(hidden_dim, hidden_dim, num_classes, num_layers=num_cls_head_layers)
            nn.init.constant_(self.class_embed.layers[-1].bias, bias_value)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes)
            nn.init.constant_(self.class_embed.bias, bias_value)
        enc_embed_binary = nn.Linear(hidden_dim, 1)
        nn.init.constant_(enc_embed_binary.bias, bias_value)

        if share_class_embed:
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(transformer.decoder.num_layers)])
            self.class_embed.append(enc_embed_binary)
        else:
            self.class_embed = _get_clones(self.class_embed, transformer.decoder.num_layers)
            self.class_embed.append(enc_embed_binary)

        
        self.transformer.decoder.class_embed = self.class_embed

        if num_reg_head_layers > 1:
            self.segment_embed = MLP(hidden_dim, hidden_dim, 2, num_layers=num_reg_head_layers)
            # self.segment_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=num_reg_head_layers)
            nn.init.zeros_(self.segment_embed.layers[-1].weight)
            nn.init.zeros_(self.segment_embed.layers[-1].bias)
        else:
            self.segment_embed = nn.Linear(hidden_dim, 2)
            # self.segment_embed = nn.Linear(hidden_dim, 4)
            nn.init.zeros_(self.segment_embed.weight)
            nn.init.zeros_(self.segment_embed.bias)

        if share_segment_embed:
            enc_segment_embed = copy.deepcopy(self.segment_embed)
            self.segment_embed = nn.ModuleList([self.segment_embed for _ in range(transformer.decoder.num_layers)])
            self.segment_embed.append(enc_segment_embed)
        else:
            self.segment_embed = _get_clones(self.segment_embed, transformer.decoder.num_layers + 1)
        self.transformer.decoder.segment_embed = self.segment_embed

    def forward(self, samples: NestedTensor, info):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            or a tuple of tensors and mask

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized segment.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)  # (n, c, t)

        # Multi scale
        srcs, masks = self.input_proj(samples.tensors, samples.mask)

        fps = torch.stack([item['fps'] for item in info if 'fps' in item])
        stride = torch.stack([item['stride'] for item in info if 'stride' in item])
        feature_durations = torch.stack([item['feature_duration'] for item in info if 'feature_duration' in item])
        srcs = [srcs[0] + self.position_embedding(fps.unsqueeze(-1))]

        grid = get_feature_grids(masks[0], fps, stride, stride)
        grids = [grid]
        poss = [self.position_embedding(grid)]
        query_embed = None

        (
            hs, inter_grids,
            enc_outputs_class, enc_outputs_coord_unact, enc_mask, enc_proposals, query_mask
        ) = self.transformer(
            srcs, masks, poss, grids,
            feature_durations, fps, stride, query_embed
        )

        outputs_classes, outputs_coords = [], []
        # gather outputs from each decoder layer
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])
            output_segments = inter_grids[lvl]

            outputs_classes.append(outputs_class)
            outputs_coords.append(output_segments)
        outputs_class = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_segments': outputs_coords[-1],
            'mask': query_mask,
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coords, query_mask)

        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class,
            'pred_segments': enc_outputs_coord_unact,
            'mask': enc_mask,
            'proposals': enc_proposals,
        }

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, query_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b, 'mask': query_mask}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DiGIT.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, diou=False, label_smoothing=0):
        """ Create the criterion.
        Parameters:
            num_classes: number of action categories, omitting the special no-action category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = 2.0
        self.label_smoothing = label_smoothing
        self.diou = diou


    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        if targets[0]['labels'].dtype == torch.long:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes_onehot = F.one_hot(target_classes_o, num_classes=src_logits.shape[2]).to(src_logits.dtype)

            target_onehot_shape = [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]]
            target_classes_onehot_full = torch.zeros(target_onehot_shape, dtype=src_logits.dtype, device=src_logits.device)
            target_classes_onehot_full[idx] = target_classes_onehot
        else:
            target_onehot_shape = [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]]
            target_classes_onehot_full = torch.zeros(target_onehot_shape, dtype=src_logits.dtype, device=src_logits.device)
            target_classes_onehot_full[idx] = torch.cat([t["labels"] for t in targets], dim=0)

        if self.label_smoothing != 0:
            target_classes_onehot_full *= 1 - self.label_smoothing
            target_classes_onehot_full += self.label_smoothing / (target_classes_onehot.size(-1) + 1)
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot_full, outputs['mask'], num_segments, alpha=self.focal_alpha, gamma=self.focal_gamma)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_segment = center_width_loss(src_segments, segment_t1t2_to_cw(target_segments))
        src_segments = torch.cat([
            src_segments[..., :1], src_segments[..., 1:].exp()
        ], dim=-1)
        if self.diou:
            loss_iou = diou_loss(segment_cw_to_t1t2(src_segments), target_segments)
        else:
            loss_iou = 1 - torch.diag(
                segment_iou(
                    segment_cw_to_t1t2(src_segments),
                    target_segments,
                )
            )

        losses = {}
        losses['loss_segments'] = loss_segment.sum() / num_segments
        losses['loss_iou'] = loss_iou.sum() / num_segments
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'segments': self.loss_segments,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        losses = {}
        # Compute all the requested losses
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']

            if enc_outputs['pred_logits'].size(-1) == 1:
                bin_targets = copy.deepcopy(targets)
                for bt in bin_targets:
                    bt['labels'] = torch.zeros_like(bt['labels'])
                enc_targets = bin_targets
            else:
                enc_targets = targets

            indices = self.matcher(enc_outputs, enc_targets, encoder=True)

            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False

                l_dict = self.get_loss(loss, enc_outputs, enc_targets, indices, num_segments, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the TADEvaluator"""

    @torch.no_grad()
    def forward(self, outputs, video_durations, feature_durations, strides, offsets, duration_thresh=0.05):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the duration of each video of the batch
        """
        out_logits, out_segments = outputs['pred_logits'], outputs['pred_segments']

        query_mask = outputs['mask'] if 'mask' in outputs else None
        assert len(out_logits) == len(video_durations)
        assert len(out_logits) == len(feature_durations)

        bs = out_logits.size(0)
        prob = out_logits.sigmoid()   # [bs, nq, C]
        if 'pred_actionness' in outputs:
            prob *= outputs['pred_actionness']
        scores, labels = prob.max(dim=-1)
        segments = torch.cat([
            out_segments[..., :1], out_segments[..., 1:].exp()
        ], dim=-1)
        segments = segment_cw_to_t1t2(segments) + offsets[:, None, None]

        # get scores of mask areas
        results = []
        for i in range(bs):
            cur_scores, cur_labels, cur_segments = scores[i], labels[i], segments[i]
            cur_segments = torch.clip(cur_segments, 0, video_durations[i].item())

            valid_mask = (cur_segments[..., 1] - cur_segments[..., 0]) > duration_thresh
            if query_mask is not query_mask:
                valid_mask = torch.logical_and(valid_mask, ~query_mask[i])
            else:
                valid_mask = valid_mask
            cur_scores, cur_labels, cur_segments = cur_scores[valid_mask], cur_labels[valid_mask], cur_segments[valid_mask]

            results.append({
                'scores': cur_scores,
                'labels': cur_labels,
                'segments': cur_segments,
            })
        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name='digit')
def build(args):
    if args.binary:
        num_classes = 1
    else:
        num_classes = args.num_classes

    pos_embed = build_position_encoding(args)
    transformer = build_deformable_transformer(args)

    model = DiGIT(
        pos_embed,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        feature_dim=args.feature_dim,
        num_sampling_levels=args.num_sampling_levels,
        num_feature_levels=args.num_feature_levels,
        kernel_size=args.kernel_size,
        num_cls_head_layers=args.num_cls_head_layers,
        num_reg_head_layers=args.num_reg_head_layers,
        aux_loss=args.aux_loss,
        with_segment_refine=False,
        emb_norm_type=args.emb_norm_type,
        emb_relu=args.emb_relu,
        fix_encoder_proposals=args.fix_encoder_proposals,
        share_class_embed=args.share_class_embed,
        share_segment_embed=args.share_segment_embed,
        emb_dropout=args.emb_dropout
    )

    matcher = build_matcher(args)
    losses = ['labels', 'segments']

    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_iou': args.iou_loss_coef,
    }

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes, matcher,
        weight_dict, losses, focal_alpha=args.focal_alpha,
        diou=args.diou, label_smoothing=args.label_smoothing,
    )

    postprocessor = PostProcess()

    return model, criterion, postprocessor
