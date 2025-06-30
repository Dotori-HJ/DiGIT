import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.

    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397

    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype)\
    .expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.

    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py

    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.

    Check pytorch's BatchNorm1d implementation for argument details.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )

    def forward(self, inp, inp_mask):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked ones.
        n = inp_mask.sum()
        mask = inp_mask / n
        mask = mask.unsqueeze(1).expand(inp.shape)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp ** 2).sum([0, 2]) - mean ** 2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp, inp_mask


class MaskedConv1D(nn.Module):
    """
    from https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/blocks.py#10
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0
        mask = ~mask.unsqueeze(1)
        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, ~out_mask.squeeze(1)


class MaskedConvTranspose1D(nn.Module):
    """
    from https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/blocks.py#10
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        # assert T % self.stride == 0
        mask = ~mask.unsqueeze(1)
        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, ~out_mask.squeeze(1)


class ConvBlock(nn.Module):
    """
    from https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/blocks.py#L735
    A simple conv block similar to the basic block used in ResNet
    """
    def __init__(
        self,
        n_channels,            # dimension of the input features
        kernel_size=3,         # conv kernel size
        stride=1,         # downsampling stride for the current layer
        expansion_factor=2,    # expansion factor of feat dims
        n_out=None,            # output dimension, if None, set to input dim
        act_layer=nn.ReLU,     # nonlinear activation used after conv, default ReLU
    ):
        super().__init__()
        # must use odd sized kernel
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_channels

         # 1x3 (strided) -> 1x3 (basic block in resnet)
        width = n_channels * expansion_factor
        self.conv1 = MaskedConv1D(
            n_channels, width, kernel_size, stride, padding=padding)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_channels, n_out, 1, stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask, pos_embd=None):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # downsampling
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # residual connection
        out += identity
        out = self.act(out)

        return out, out_mask


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x, mask):
        out = super().forward(x)
        return out, mask


class LayerNorm(nn.Module):
    """
    from https://github.com/happyharrycn/actionformer_release/libs/modeling/blocks.py#L63
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x, mask):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out, mask


class Sequential(nn.Sequential):
    def forward(self, x, mask):
        for module in self:
            x, mask = module(x, mask)
        return x, mask


class Identity(nn.Identity):
    def forward(self, x, mask):
        out = super().forward(x)
        return out, mask


class Dropout(nn.Dropout):
    def forward(self, x, mask):
        out = super().forward(x)
        return out, mask


class ReLU(nn.ReLU):
    def forward(self, x, mask):
        out = super().forward(x)
        return out, mask


class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        feature_dim,               # input feature dimension
        hidden_dim,             # embedding dimension (after convolution)
        kernel_size,          # conv kernel size of the embedding network
        arch=(2, 2),   # (#convs, #stem convs, #branch convs)
        num_feature_levels=5,
        scale_factor=2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
        dropout=0.0,
    ):
        super().__init__()
        # assert num_feature_levels > 1
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.arch = arch
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # embedding network using convs
        self.embd = nn.ModuleList()
        norm_layer = LayerNorm
        for idx in range(arch[0]):
            feature_dim = hidden_dim if idx > 0 else feature_dim
            self.embd.append(
                Sequential(
                    MaskedConv1D(
                        feature_dim, hidden_dim, kernel_size,
                        stride=1, padding=kernel_size//2, bias=False
                    ),
                    norm_layer(hidden_dim),
                    Dropout(dropout),
                )
            )

        for layer in self.embd:
            nn.init.xavier_uniform_(layer[0].conv.weight, gain=1)
            # nn.init.zeros_(layer[0].conv.bias)

        # stem network using convs
        self.stem = nn.ModuleList([
            ConvBlock(hidden_dim, kernel_size=3, stride=1)
            for _ in range(arch[1])
        ])
        self.norm = LayerNorm(hidden_dim)

        # main branch using convs with pooling
        self.branch = nn.ModuleList([
            Sequential(
                MaskedConv1D(
                    hidden_dim, hidden_dim, kernel_size=3,
                    stride=self.scale_factor, padding=1
                ),
                norm_layer(hidden_dim),
                Dropout(dropout),
            )
            for _ in range(num_feature_levels-1)
        ])
        for layer in self.branch:
            nn.init.xavier_uniform_(layer[0].conv.weight, gain=1)
            nn.init.zeros_(layer[0].conv.bias)
        # init weights
        self.apply(self.__init_weights__)


    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            m = ~mask.unsqueeze(1)
            x = x * m.detach()

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # x, mask = self.norm(x, mask)
        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks


class MaskedConvTranspose1D(nn.Module):
    """
    from https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/blocks.py#10
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        # assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=bias, padding_mode=padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        # assert T % self.stride == 0
        mask = ~mask.unsqueeze(1)
        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, ~out_mask.squeeze(1)


class FPN1D(nn.Module):
    """
        Feature pyramid network
    """
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True,     # if to apply layer norm at the end
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # disable bias if using layer norm
            l_conv = MaskedConv1D(
                in_channels[i], out_channel, 1, bias=(not with_ln)
            )
            # use depthwise conv here for efficiency
            fpn_conv = MaskedConv1D(
                out_channel, out_channel, 3,
                padding=1, groups=out_channel
            )
            # fpn_conv = ConvBlock(out_channel, 3, stride=1)
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # build laterals, fpn_masks will remain the same with 1x1 convs
        laterals = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level]
            )
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=self.scale_factor, mode='nearest'
            )

        # fpn conv / norm -> outputs
        # mask will remain the same
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(used_backbone_levels):
            x, new_mask = self.fpn_convs[i](
                laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x, new_mask)[0]
            fpn_feats += (x, )
            new_fpn_masks += (new_mask, )

        return fpn_feats, new_fpn_masks


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)

    def forward(self, x, mask):
        out = super().forward(x)
        return out, mask

class MaskedResizer(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()

    def forward(self, x, mask):
        mask_out = F.interpolate(mask.unsqueeze(1).float(), scale_factor=0.5, mode='linear').squeeze(1) > 0.5
        out = torch.zeros(x.size(0), x.size(1), mask_out.size(1), device=x.device)

        for i in range(x.size(0)):
            valid_len = (~mask[i]).sum().item()
            out_x = F.interpolate(x[[i], :, :valid_len], size=(~mask_out[i]).sum().item(), mode='linear')
            out[i, :, :out_x.size(2)] = out_x[0]

        return out, mask_out


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()

        # classifier
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size//2
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(Identity())

        # segment regression
        self.offset_head = MaskedConv1D(
            feat_dim, 2, kernel_size,
            stride=1, padding=kernel_size//2
        )
        # TODO: bias init for offset head

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out, cur_mask)[0])
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (cur_offsets, )

        # fpn_masks remains the same
        return out_offsets


class EncoderMultiScaleProj(nn.Module):
    """
        A backbone that with only conv
        from https://github.com/happyharrycn/actionformer_release/libs/modeling/backbones.py#L168
    """
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        kernel_size,
        num_feature_levels=5,
        scale_factor=0.5,
        with_ln=False,
    ):
        super().__init__()
        # assert num_feature_levels > 1
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        kernel_sizes = [1, 3, 5, 7, 9, 11]

        # main branch using convs with pooling
        self.embedders = nn.ModuleList([
            Sequential(
                MaskedConv1D(feature_dim, hidden_dim, kernel_size=kernel_sizes[i], padding=kernel_sizes[i]//2, bias=with_ln),
                LayerNorm(hidden_dim) if with_ln else Identity(),
            )
            for i in range(num_feature_levels)
        ])
        for layer in self.embedders:
            nn.init.xavier_uniform_(layer[0].conv.weight, gain=1)
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        # Resize features
        x = (~mask).unsqueeze(1).float().detach() * x

        # Embed features
        out_feats, out_masks = [], []
        for i, embedder in enumerate(self.embedders):
            cur_feat, cur_mask = embedder(x, mask)
            out_feats += (cur_feat, )
            out_masks += (cur_mask, )

        return out_feats, out_masks


class ResizerBackbone(nn.Module):
    """
        A backbone that with only conv
        from https://github.com/happyharrycn/actionformer_release/libs/modeling/backbones.py#L168
    """
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        kernel_size,
        num_feature_levels=5,
        scale_factor=0.5,
        with_ln=False,
    ):
        super().__init__()
        # assert num_feature_levels > 1
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        self.resizers = nn.ModuleList([
            MaskedResizer(scale_factor)
            for i in range(1, num_feature_levels)
        ])

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        # Resize features
        x = (~mask).unsqueeze(1).float().detach() * x
        resized_feats = (x, )
        resized_masks = (mask, )
        cur_feat, cur_mask = x, mask
        for resizer in self.resizers:
            cur_feat, cur_mask = resizer(cur_feat, cur_mask)
            resized_feats += (cur_feat, )
            resized_masks += (cur_mask, )

        return resized_feats, resized_masks


class MultiScaleFPN(nn.Module):
    """
        A backbone that with only conv
        from https://github.com/happyharrycn/actionformer_release/libs/modeling/backbones.py#L168
    """
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        kernel_size,
        num_feature_levels=1,
        scale_factor=0.5,
        with_ln=True,
    ):
        super().__init__()
        # assert num_feature_levels > 1
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        self.resizers = nn.ModuleList([
            MaskedResizer(scale_factor)
            for i in range(1, num_feature_levels)
        ])

        self.laterals1 = nn.ModuleList([
            Sequential(
                MaskedConv1D(feature_dim, hidden_dim, kernel_size=3, stride=1,padding=1),
                LayerNorm(hidden_dim) if with_ln else Identity(),
            )
            for _ in range(num_feature_levels)
        ])

        self.laterals2 = nn.ModuleList([
            Sequential(
                MaskedConv1D(feature_dim, hidden_dim, kernel_size=3, stride=1,padding=1),
                LayerNorm(hidden_dim) if with_ln else Identity(),
            )
            for _ in range(num_feature_levels)
        ])


        # main branch using convs with pooling
        self.fpn_outs = nn.ModuleList([
            Sequential(
                MaskedConv1D(feature_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=with_ln),
                LayerNorm(hidden_dim) if with_ln else Identity(),
            )
            for _ in range(num_feature_levels)
        ])
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        for layer in self.embedders:
            nn.init.xavier_uniform_(layer[0].conv.weight, gain=1)
            # nn.init.constant_(layer[0].conv.bias, bias_value)
        #     if not with_ln:
        #         nn.init.zeros_(layer[0].conv.bias)
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        # Resize features
        x = (~mask).unsqueeze(1).float().detach() * x
        resized_feats = (x, )
        resized_masks = (mask, )
        cur_feat, cur_mask = x, mask
        for resizer in self.resizers:
            cur_feat, cur_mask = resizer(cur_feat, cur_mask)
            resized_feats += (cur_feat, )
            resized_masks += (cur_mask, )

        # Embed features
        out_feats, out_masks = [], []
        for i, embedder in enumerate(self.embedders):
            cur_feat, cur_mask = embedder(resized_feats[i], resized_masks[i])
            out_feats += (cur_feat, )
            out_masks += (cur_mask, )

        return out_feats, out_masks


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        bias=True, kernel_size=3,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.dwconv = nn.Conv1d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=kernel_size//2, groups=med_channels, bias=bias
        ) # depthwise conv
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x, mask):
        m = (~mask).float().unsqueeze(1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = x * m
        x = self.dwconv(x)
        x = x * m
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.pwconv2(x)
        return x


class ConvAttn(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, kernel_size=3, expansion_factor=2, bias=True, **kwargs, ):
        super().__init__()
        width = int(dim * expansion_factor)
        self.conv1 = nn.Conv1d(dim, width, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.conv2 = nn.Conv1d(width, dim, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        m = (~mask).float().unsqueeze(1).detach()
        x = x.permute(0, 2, 1)
        x = x * m
        x = self.conv1(x)
        x = self.act(x)
        x = x * m
        x = self.conv2(x)
        x = x * m
        x = x.permute(0, 2, 1)
        return x


class GatedConv(nn.Module):
    def __init__(self, dim, kernel_size=7, num_levels=4, hidden_dim=256, bias=True, group_conv=True, **kwargs, ):
        super().__init__()
        assert hidden_dim % num_levels == 0
        self.dim = dim
        self.num_levels = num_levels
        self.bias = bias
        self.hidden_dim = hidden_dim
        self.conv_dim = hidden_dim // num_levels
        convs = []
        for i in range(self.num_levels):
            convs.append(
                nn.Conv1d(
                    self.conv_dim,
                    self.conv_dim,
                    kernel_size=kernel_size,
                    padding=(i+1)*(kernel_size//2),
                    bias=bias,
                    dilation=i+1,
                    groups=self.conv_dim if group_conv else 1,
                )
            )
        self.convs = nn.ModuleList(convs)
        self.fc_in = nn.Linear(dim, self.hidden_dim * 2)
        self.split_sizes = [self.hidden_dim] * 2
        self.fc_out = nn.Linear(self.hidden_dim, dim)
        self.act = nn.SiLU()

    def forward(self, x, mask=None):
        conv_path, gate_path = torch.split(self.fc_in(x), self.split_sizes, dim=-1)
        if mask is not None:
            m = (~mask).float().unsqueeze(1).detach()
            conv_path = conv_path.transpose(1, 2) * m
            conv_outs = [
                conv(conv_in)
                for conv, conv_in in zip(self.convs, torch.split(conv_path, self.conv_dim, dim=1))
            ]
            conv_outs = torch.cat(conv_outs, dim=1) * m
        else:
            conv_path = conv_path.transpose(1, 2)
            conv_outs = [
                conv(conv_in)
                for conv, conv_in in zip(self.convs, torch.split(conv_path, self.hidden_dim, dim=1))
            ]
            conv_outs = torch.cat(conv_outs, dim=1)
        conv_outs = conv_outs.transpose(1, 2)
        
        return self.fc_out(conv_outs * self.act(gate_path))


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


if __name__ == '__main__':
    backbone = ConvBackbone(
        2048,
        256,
        kernel_size=3,
        arch=(2, 2),
        scale_factor=2,
        with_ln=True,
    )
    import torch
    x = torch.randn(1, 2048, 128)
    x, mask = backbone(x, torch.zeros(1, 128, dtype=torch.bool))
    print(len(x), len(mask), [i.size() for i in x], [m.size() for m in mask])