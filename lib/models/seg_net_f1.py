import torch.nn as nn
import torchvision.models as models
from lib.models.seg_net import SegNet, SegNetDown2, SegNetDown3, SegNetUp2, \
    SegNetUp3


class SegNetF1(SegNet):
    def __init__(self,
                 n_classes=21,
                 in_channels=3,
                 is_unpooling=True,
                 init_vgg16_params=True):
        super(SegNet, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = SegNetDown2(self.in_channels, 64)
        self.down2 = SegNetDown2(64, 128)
        self.down3 = SegNetDown3(128, 256)
        self.down4 = SegNetDown3(256, 512)
        self.down5 = SegNetDown3(512, 512)

        self.up5 = SegNetUp3(512, 512)
        self.up4 = SegNetUp3(512, 256)
        self.up3 = SegNetUp3(256, 128)
        self.up2 = SegNetUp2(128, 64)
        self.up1 = SegNetUp2(64, n_classes)

        self.fc = nn.Linear(4362240, 1)

        if init_vgg16_params:
            vgg16 = models.vgg16(pretrained=True)
            self.init_vgg16_params(vgg16)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        up2 = up2.view(up2.size(0), -1)
        f1 = self.fc(up2)

        return up1, f1


def build_seg_net_f1(n_classes=21):
    net = SegNetF1(n_classes=n_classes)
    return net