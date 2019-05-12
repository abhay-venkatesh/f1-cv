import torch


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