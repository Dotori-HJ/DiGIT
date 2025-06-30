# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn.functional as F
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .utils import smooth_l1_loss, sigmoid_focal_loss
from util.segment_ops import segment_cw_to_t1t2, pairwise_segment_diou, segment_t1t2_to_cw, segment_iou, pairwise_log_ratio_width_loss


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_seg: float = 1, cost_iou: float = 1, diou=False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_seg: This is the relative weight of the L1 error of the segment coordinates in the matching cost
            cost_iou: This is the relative weight of the iou loss of the segment in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_seg = cost_seg
        self.cost_iou = cost_iou
        self.diou = diou
        assert cost_class != 0 or cost_seg!= 0 or cost_iou != 0, "all costs cant be 0"

    # def add_class_cost(self):
    #     self.cost_class_ = self.cost_class

    @torch.no_grad()
    def forward(self, outputs, targets, encoder=False):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_segments": Tensor of dim [batch_size, num_queries, 2] with the predicted segment coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_segments] (where num_target_segments is the number of ground-truth
                           objects in the target) containing the class labels
                 "segments": Tensor of dim [num_target_segments, 2] containing the target segment coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_segments)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  #  [batch_size * num_queries, num_classes]
        out_seg = outputs["pred_segments"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and segments
        tgt_ids = torch.cat([v["labels"] for v in targets])  # shape = n1+n2+...
        tgt_seg = torch.cat([v["segments"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]


        cost_seg = pairwise_log_ratio_width_loss(out_seg, segment_t1t2_to_cw(tgt_seg))

        # Compute the iou cost betwen segments
        out_seg = torch.cat([
            out_seg[..., :1], out_seg[..., 1:].exp()
        ], dim=-1)
        if self.diou:
            cost_iou = -pairwise_segment_diou(segment_cw_to_t1t2(out_seg), tgt_seg)
        else:
            cost_iou = -segment_iou(segment_cw_to_t1t2(out_seg), tgt_seg)

        # Final cost matrix, [bs x nq, batch_ngt]
        if encoder:
            C = self.cost_seg * cost_seg + self.cost_iou * cost_iou + cost_class
        else:
            C = self.cost_class * cost_class + self.cost_iou * cost_iou + self.cost_seg * cost_seg
        if outputs['mask'] is not None:
            C = C.masked_fill(outputs['mask'].flatten(0, 1).unsqueeze(-1), 1e6)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["segments"]) for v in targets]
        if C.isnan().any():
            print(C.shape)
            print(C.isnan().sum(), cost_class.isnan().sum(), cost_iou.isnan().sum())
        try:
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        except Exception as e:
            print(e)
            print(C)
            exit()
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_seg=args.set_cost_seg, cost_iou=args.set_cost_iou, diou=args.diou)