# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet

from .resnet_optim import ResNet_Optim, ResNetV1d_Optim
from .resnext_optim import ResNeXt_Optim

from .dwt_mobilev2 import DWT_MobileNetV2
from .dwt_efficientnet import DWTEfficientNet
from .efficientnet_lite import EfficientNetLite
from .att_efficientnet_lite import AttEfficientNetLite
from .eca_efficientnet import ECAEfficientNet

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet',
    'ResNet_Optim', 'ResNeXt_Optim', 'DWT_MobileNetV2', 'DWTEfficientNet',
    'EfficientNetLite', 'AttEfficientNetLite', 'ECAEfficientNet'
]
