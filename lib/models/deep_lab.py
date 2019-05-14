from lib.models.deeplab.aspp import build_aspp
from lib.models.deeplab.backbone import build_backbone
from lib.models.deeplab.decoder import build_decoder
from lib.models.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from lib.models.deeplab.sync_batchnorm.replicate import \
    patch_replication_callback
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLab(nn.Module):
    """ Reference: jfzhang95/pytorch-deeplab-xception """

    def __init__(self,
                 backbone='resnet',
                 output_stride=16,
                 n_classes=21,
                 sync_bn=True,
                 freeze_bn=False):
        super(DeepLab, self).__init__()
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

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(
            x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (isinstance(m[1], nn.Conv2d)
                        or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (isinstance(m[1], nn.Conv2d)
                        or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class DeepLabF1(DeepLab):
    def __init__(self,
                 backbone='resnet',
                 output_stride=16,
                 n_classes=21,
                 sync_bn=True,
                 freeze_bn=False):
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

        self.fc = nn.Linear(99360, 1)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        f1 = self.fc(x.view(x.size(0), -1))
        x = F.interpolate(
            x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x, f1


def build_deep_lab(n_classes=21):
    net = DeepLab(n_classes=n_classes)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    return net


def build_deep_lab_f1(n_classes=21):
    net = DeepLabF1(n_classes=n_classes)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    return net
