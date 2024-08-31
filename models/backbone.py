# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class FPNModule(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPNModule, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                self.inner_blocks.append(None)
                self.layer_blocks.append(None)
            else:
                self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
                self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1))

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = [last_inner]  # 마지막 레이어의 출력을 추가
        for i in range(len(x) - 2, -1, -1):
            if self.inner_blocks[i] is None:
                continue
            inner_lateral = self.inner_blocks[i](x[i])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[i](last_inner))
        return results

class ASFFLayer(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(ASFFLayer, self).__init__()
        self.asff_modules = nn.ModuleList([self._make_asff_layer(in_c, out_channels) for in_c in in_channels_list])

    def _make_asff_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        # 입력 리스트의 개수를 확인하고, ASFF 레이어의 모듈 리스트와 비교합니다.
        assert len(inputs) == len(self.asff_modules), f"Expected {len(self.asff_modules)} inputs, but got {len(inputs)}"
        fused_outs = [self.asff_modules[i](inputs[i]) for i in range(len(inputs))]
        return sum(fused_outs)

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

        # FPN과 ASFF 모듈 추가
        self.fpn = FPNModule(in_channels_list=[256, 512, 1024, 2048], out_channels=num_channels)
        self.asff = ASFFLayer(in_channels_list=[num_channels] * len(return_layers), out_channels=num_channels)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        
        # NestedTensor에서 텐서를 추출
        features = [out[name].tensors for name in out.keys()]

        # FPN을 통해 다중 스케일 피처 생성
        fpn_features = self.fpn(features)
        
        # ASFF를 통해 피처 통합
        integrated_features = self.asff(fpn_features)

        # 통합된 피처를 NestedTensor 형태로 반환
        final_out = NestedTensor(integrated_features, mask)

        return final_out




class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        
        out = []
        pos = []
        
        # xs가 dict가 아니라 NestedTensor일 경우 처리
        if isinstance(xs, NestedTensor):
            out.append(xs)
            pos.append(self[1](xs).to(xs.tensors.dtype))
        else:
            for name, x in xs.items():
                out.append(x)
                # position encoding
                pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos




def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
