import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16

from ..builder import NECKS
from ..utils import MemoryEfficientSwish, SeparableConv2d
from .bifpn import WeightedMerge, Resample


@NECKS.register_module()
class NewPaFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 target_size_list,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 conv_cfg=None,
                 norm_cfg=dict(
                     type='BN', momentum=0.003, eps=1e-4, requires_grad=True)):
        super(NewPaFPN, self).__init__()
        assert len(in_channels) >= 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.stack = stack
        self.num_outs = num_outs
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        # add extra feature layers using resampling
        self.extra_ops = nn.ModuleList()
        for i in range(self.backbone_end_level, self.num_outs):
            in_c = in_channels[-1]
            self.extra_ops.append(
                Resample(
                    in_c,
                    out_channels,
                    target_size_list[i],
                    norm_cfg,
                    apply_bn=True))
            in_channels.append(out_channels)

        self.stack_bifpns = nn.ModuleList()
        for _ in range(stack):
            self.stack_bifpns.append(
                PaFPNLayer(
                    in_channels,
                    out_channels,
                    target_size_list,
                    num_outs=self.num_outs,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            in_channels = [out_channels] * self.num_outs

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SeparableConv2d):
                m.init_weights()

    @auto_fp16()
    def forward(self, inputs):
        outs = list(inputs)
        for _, extra_op in enumerate(self.extra_ops):
            outs.append(extra_op(outs[-1]))

        for _, stack_bifpn in enumerate(self.stack_bifpns):
            outs = stack_bifpn(outs)

        return tuple(outs[:self.num_outs])


class PaFPNLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 target_size_list,
                 num_outs=5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(PaFPNLayer, self).__init__()
        assert num_outs >= 5 and (num_outs - 5) % 2 == 0
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.num_outs = num_outs

        self.top_down_merge = nn.ModuleList()
        for i in range(self.num_outs - 1, 0, -1):
            if i == self.num_outs - 1:
                in_channels_list = [in_channels[i], in_channels[i - 1]]
            elif (self.num_outs - i - 1) % 2 == 0:
                in_channels_list = [
                    out_channels, in_channels[i - 1], out_channels
                ]
            else:
                in_channels_list = [out_channels, in_channels[i - 1]]

            merge_op = WeightedMerge(
                in_channels_list,
                out_channels,
                target_size_list[i - 1],
                norm_cfg,
                apply_bn=True)
            self.top_down_merge.append(merge_op)

        self.bottom_up_merge = nn.ModuleList()
        for i in range(0, self.num_outs - 1):
            if i == self.num_outs - 2:
                in_channels_list = [
                    in_channels[-1], out_channels, out_channels
                ]
            elif (i + 1) % 2 == 0:
                in_channels_list = [out_channels, out_channels, out_channels]
            else:
                in_channels_list = [out_channels, out_channels]

            merge_op = WeightedMerge(
                in_channels_list,
                out_channels,
                target_size_list[i + 1],
                norm_cfg,
                apply_bn=True)
            self.bottom_up_merge.append(merge_op)

    def forward(self, inputs):
        assert len(inputs) == self.num_outs

        # top down merge
        md_x = []
        for i in range(self.num_outs - 1, 0, -1):
            if i == self.num_outs - 1:
                inputs_list = [inputs[i], inputs[i - 1]]
            elif (self.num_outs - i - 1) % 2 == 0:
                inputs_list = [md_x[-1], inputs[i - 1], md_x[-2]]
            else:
                inputs_list = [md_x[-1], inputs[i - 1]]
            x = self.top_down_merge[self.num_outs - i - 1](inputs_list)
            md_x.append(x)

        # bottom up merge
        outputs = md_x[::-1]
        for i in range(1, self.num_outs - 1):
            inputs_list = [outputs[i], outputs[i - 1]]
            if i % 2 == 0:
                inputs_list.append(outputs[i - 2])
            outputs[i] = self.bottom_up_merge[i - 1](inputs_list)
        outputs.append(
            self.bottom_up_merge[-1]([inputs[-1], outputs[-1], outputs[-2]]))
        return outputs
