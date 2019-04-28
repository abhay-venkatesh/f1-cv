import torch
import torch.nn.functional as F


def cross_entropy2d(output, target, weight=None):
    n, c, h, w = output.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between output and target
    if h != ht and w != wt:  # upsample labels
        output = F.interpolate(
            output, size=(ht, wt), mode="bilinear", align_corners=True)

    output = output.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(output, target, weight=weight)

    return loss


def get_iou(outputs, labels):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most
    # probably be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1,
                                            2))  # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH
                                     )  # We smooth our devision to avoid 0/0
    thresholded = torch.clamp(
        20 * (iou - 0.5), 0,
        10).ceil() / 10  # This is equal to comparing with thresolds
    return thresholded.mean()


def lagrange(num_pos, y_, y, w, eps, tau, lamb, mu, gamma, device):
    y = y.float()
    y_ = y_.squeeze()

    neutral = (num_pos * eps)

    neg = torch.max(
        torch.zeros(y_.shape, dtype=torch.float, device=device),
        eps + (w * y_))
    neg = (abs(1 - y) * neg).sum()
    neg /= (len(y) - y.sum())

    pos = mu * ((y * tau).sum() - 1)
    pos += (y * lamb * (tau - (w * y_))).sum()
    pos += gamma * ((y * tau).sum() - eps)
    pos /= y.sum()
    return neutral + neg + pos