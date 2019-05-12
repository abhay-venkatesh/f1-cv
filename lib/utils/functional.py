import torch
import torch.nn.functional as F


def cross_entropy2d(output, target, weight=None):
    # Parallel loss computation
    if torch.cuda.device_count() > 1:
        loss = CriterionParallel(cross_entropy2d)

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


def lagrange(num_pos, y1_, y1, w, eps, tau, lamb, mu, gamma):
    # Reshape
    y1 = y1.float()
    y1_ = y1_.squeeze()

    # Term that is not associated with either positive or negative examples
    neutral = (num_pos * eps)

    # Negative example terms
    neg = torch.max(torch.zeros_like(y1_), eps + (w * y1_))
    neg = (abs(1 - y1) * neg).sum()

    # Positive example terms
    pos = mu * ((y1 * tau).sum() - 1)
    pos += (y1 * lamb * (tau - (w * y1_))).sum()
    pos += (y1 * gamma * (tau - eps)).sum()

    return neutral + neg + pos


def partial_lagrange(num_pos, y1_, y1, w, eps, tau, lamb, mu, gamma):
    # Reshape
    y1 = y1.float()
    y1_ = y1_.squeeze()

    # Term that is not associated with either positive or negative examples
    neutral = (num_pos * eps)

    # Negative example terms
    neg = torch.max(torch.zeros_like(y1_), eps + (w * y1_))
    neg = (abs(1 - y1) * neg).sum()

    # Positive example terms
    pos = mu * ((y1 * tau).sum() - 1)
    pos += (y1 * lamb * (tau - (w * y1_))).sum()

    return neutral + neg + pos


def sorted_project(eps, tau):
    tau_sorted, indices = tau.clone().detach().cuda().sort(descending=True)
    new_eps = eps.clone().detach().cuda()
    dataset_size = len(tau)
    k = dataset_size + 1
    for i in range(dataset_size):
        if tau_sorted[i] <= max(new_eps, 0):
            k = i
            break
        else:
            new_eps = (((i * new_eps) + tau_sorted[i]) / (i + 1))
    new_eps = torch.max(eps, torch.zeros_like(eps))
    new_tau = torch.full_like(tau_sorted, 0)
    new_tau[:k - 1] = new_eps
    new_tau[k:] = tau_sorted[k:]
    new_tau[indices] = new_tau

    # Return
    eps.data = new_eps
    tau.data = new_tau


def naive_project(eps, tau, I):
    if eps < 0:
        eps.data = torch.full_like(eps, 0)
    else:
        dummy = tau.clone().detach().cuda()
        for i in I:
            if tau[i] > eps:
                dummy[i] = (tau[i] + eps) / 2
                eps.data = dummy[i]
        tau.data = dummy


class ModularizedFunction(torch.nn.Module):
    """
    A Module which calls the specified function in place of the forward pass.
    Useful when your existing loss is functional and you need it to be a
     Module.
    """

    def __init__(self, forward_op):
        super().__init__()
        self.forward_op = forward_op

    def forward(self, *args, **kwargs):
        return self.forward_op(*args, **kwargs)


class CriterionParallel(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        if not isinstance(criterion, torch.nn.Module):
            criterion = ModularizedFunction(criterion)
        self.criterion = torch.nn.DataParallel(criterion)

    def forward(self, *args, **kwargs):
        """
        Note the .mean() here, which is required since
         DataParallel gathers any scalar outputs of forward() into a vector
         with one item per GPU (See DataParallel docs).
        """
        return self.criterion(*args, **kwargs).mean()