# ------------------------------------------------------------------------
# Modified from TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.#
#  ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Positional encodings for the transformer.
"""
import math

import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on videos.
    """
    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)  # N x T
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # N x T x C
        # n,c,t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = pos_x.permute(0, 2, 1)    # N x C x T
        return pos


class TimeBasedPositionEmbeddingSine(nn.Module):
    """Positional encoding layer using sine and cosine functions.

    Args:
    d_model (int): Dimension of the positional encoding (usually same as model's hidden dimension).
    temperature (float): Temperature parameter to scale the positional encodings.

    """
    def __init__(self, d_model=256, temperature=10000):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        # self.div_term = torch.exp(
        #     torch.arange(0., self.d_model, 2) *
        #     -(torch.log(torch.tensor(temperature, dtype=torch.float32)) / self.d_model)
        # )[None, None, :]

    def forward(self, grids):
        """Computes the positional encoding for the given center_times.

        Args:
        center_times (torch.Tensor): Tensor containing the center times.

        Returns:
        torch.Tensor: Tensor with positional encodings.

        """
        num_pos_feats = self.d_model
        dim_t = torch.arange(self.d_model, dtype=torch.float32, device=grids.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)
        # N, L, 4
        # N, L, 4, 128
        pos = grids[:, :, None, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)

        return pos.transpose(2, 1)


def build_position_encoding(args):
    feat_dim = args.hidden_dim
    position_embedding = TimeBasedPositionEmbeddingSine(feat_dim, args.temperature)

    return position_embedding
