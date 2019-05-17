from lib.models.deeplab.aspp import build_aspp
from lib.models.deeplab.backbone import build_backbone
from lib.models.deeplab.decoder import build_decoder
from lib.models.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from lib.models.deeplab.sync_batchnorm.replicate import \
    patch_replication_callback
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.deep_lab import DeepLab
import math


class DeepLabF1(DeepLab):
    def __init__(self,
                 backbone='resnet',
                 output_stride=16,
                 n_classes=21,
                 sync_bn=True,
                 freeze_bn=False,
                 size=(321, 321)):
        super(DeepLabF1, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn is True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(n_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

        self.fc = nn.Linear((math.ceil(size[0]/4)*math.ceil(size[1]/4))*92, 1)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        f1 = self.fc(x.view(x.size(0), -1))
        x = F.interpolate(
            x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x, f1


def build_deep_lab_f1(n_classes=21, size=(321, 321)):
    net = DeepLabF1(n_classes=n_classes, size=size)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    return net
