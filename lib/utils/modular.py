import torch
import torch.nn.functional as F


class CrossEntropy2D(torch.nn.Module):
    def __init__(self):
        super(CrossEntropy2D, self).__init__()

    def forward(self, output, target, weight=None):
        n, c, h, w = output.size()

        nt, ht, wt = target.size()

        # Handle inconsistent size between output and target
        if h != ht and w != wt:  # upsample labels
            output = F.interpolate(
                output, size=(ht, wt), mode="bilinear", align_corners=True)

        output = output.transpose(1, 2).transpose(2, 3).contiguous().view(
            -1, c)
        target = target.view(-1)
        loss = F.cross_entropy(output, target, weight=weight)

        return loss


class CriterionParallel(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = torch.nn.DataParallel(criterion)

    def forward(self, *args, **kwargs):
        """ Note the .mean() here, which is required since DataParallel gathers
         any scalar outputs of forward() into a vector with one item per GPU
          (See DataParallel docs). """
        return self.criterion(*args, **kwargs).mean()