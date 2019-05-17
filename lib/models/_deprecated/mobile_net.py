from lib.models.deep_lab import DeepLab, DeepLabF1


def build_mobile_net(n_classes=2):
    return DeepLab(backbone="mobilenet", n_classes=n_classes)


def build_mobile_net_f1(n_classes=2):
    return DeepLabF1(backbone="mobilenet", n_classes=n_classes)