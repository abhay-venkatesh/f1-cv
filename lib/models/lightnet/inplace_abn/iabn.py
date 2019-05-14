from os import path
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load
import torch
import torch.autograd as autograd
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional

_src_path = path.join(path.dirname(path.abspath(__file__)), "src")
_backend = load(
    name="inplace_abn",
    extra_cflags=["-O3"],
    sources=[
        path.join(_src_path, f) for f in [
            "inplace_abn.cpp", "inplace_abn_cpu.cpp", "inplace_abn_cuda.cu",
            "inplace_abn_cuda_half.cu"
        ]
    ],
    extra_cuda_cflags=["--expt-extended-lambda"])

# Activation names
ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"


def _check(fn, *args, **kwargs):
    success = fn(*args, **kwargs)
    if not success:
        raise RuntimeError("CUDA Error encountered in {}".format(fn))


def _broadcast_shape(x):
    out_size = []
    for i, s in enumerate(x.size()):
        if i != 1:
            out_size.append(1)
        else:
            out_size.append(s)
    return out_size


def _reduce(x):
    if len(x.size()) == 2:
        return x.sum(dim=0)
    else:
        n, c = x.size()[0:2]
        return x.contiguous().view((n, c, -1)).sum(2).sum(0)


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_forward(x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_forward(x)
    elif ctx.activation == ACT_NONE:
        pass


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_backward(x, dx, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_backward(x, dx)
    elif ctx.activation == ACT_NONE:
        pass


class InPlaceABN(autograd.Function):
    @staticmethod
    def forward(ctx,
                x,
                weight,
                bias,
                running_mean,
                running_var,
                training=True,
                momentum=0.1,
                eps=1e-05,
                activation=ACT_LEAKY_RELU,
                slope=0.01):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None

        # Prepare inputs
        count = _count_samples(x)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(
            0, dtype=torch.float32)
        bias = bias.contiguous() if ctx.affine else x.new_empty(
            0, dtype=torch.float32)

        if ctx.training:
            mean, var = _backend.mean_var(x)

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(
                ctx.momentum * var * count / (count - 1))

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)

        # BN forward + activation
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine,
                                          ctx.eps)
        else:
            # TODO: implement simplified CUDA backward for inference mode
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))

        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine,
                               ctx.eps)
        # dweight = eydz * weight.sign() if ctx.affine else None
        dweight = eydz if ctx.affine else None
        if dweight is not None:
            dweight[weight < 0] *= -1
        dbias = edz if ctx.affine else None

        return dx, dweight, dbias, None, None, None, None, None, None, None


class InPlaceABNSync(autograd.Function):
    @classmethod
    def forward(cls,
                ctx,
                x,
                weight,
                bias,
                running_mean,
                running_var,
                training=True,
                momentum=0.1,
                eps=1e-05,
                activation=ACT_LEAKY_RELU,
                slope=0.01,
                equal_batches=True):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None

        # Prepare inputs
        ctx.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # count = _count_samples(x)
        batch_size = x.new_tensor([x.shape[0]], dtype=torch.long)

        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(
            0, dtype=torch.float32)
        bias = bias.contiguous() if ctx.affine else x.new_empty(
            0, dtype=torch.float32)

        if ctx.training:
            mean, var = _backend.mean_var(x)
            if ctx.world_size > 1:
                # get global batch size
                if equal_batches:
                    batch_size *= ctx.world_size
                else:
                    dist.all_reduce(batch_size, dist.ReduceOp.SUM)

                ctx.factor = x.shape[0] / float(batch_size.item())

                mean_all = mean.clone() * ctx.factor
                dist.all_reduce(mean_all, dist.ReduceOp.SUM)

                var_all = (var + (mean - mean_all)**2) * ctx.factor
                dist.all_reduce(var_all, dist.ReduceOp.SUM)

                mean = mean_all
                var = var_all

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            count = batch_size.item() * x.view(x.shape[0], x.shape[1],
                                               -1).shape[-1]
            running_var.mul_((1 - ctx.momentum)).add_(
                ctx.momentum * var * (float(count) / (count - 1)))

            # Mark in-place modified tensors
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)

        # BN forward + activation
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine,
                                          ctx.eps)
            edz_local = edz.clone()
            eydz_local = eydz.clone()

            if ctx.world_size > 1:
                edz *= ctx.factor
                dist.all_reduce(edz, dist.ReduceOp.SUM)

                eydz *= ctx.factor
                dist.all_reduce(eydz, dist.ReduceOp.SUM)
        else:
            edz_local = edz = dz.new_zeros(dz.size(1))
            eydz_local = eydz = dz.new_zeros(dz.size(1))

        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine,
                               ctx.eps)
        # dweight = eydz_local * weight.sign() if ctx.affine else None
        dweight = eydz_local if ctx.affine else None
        if dweight is not None:
            dweight[weight < 0] *= -1
        dbias = edz_local if ctx.affine else None

        return dx, dweight, dbias, None, None, None, None, None, None, None


inplace_abn = InPlaceABN.apply
inplace_abn_sync = InPlaceABNSync.apply

__all__ = [
    "inplace_abn", "inplace_abn_sync", "ACT_RELU", "ACT_LEAKY_RELU", "ACT_ELU",
    "ACT_NONE"
]


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 activation="leaky_relu",
                 slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after
             normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or
             `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.batch_norm(x, self.running_mean, self.running_var,
                                  self.weight, self.bias, self.training,
                                  self.momentum, self.eps)

        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(
                x, negative_slope=self.slope, inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class InPlaceABN(ABN):
    """InPlace Activated Batch Normalization"""

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 activation="leaky_relu",
                 slope=0.01):
        """Creates an InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after
             normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or
             `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABN, self).__init__(num_features, eps, momentum, affine,
                                         activation, slope)

    def forward(self, x):
        return inplace_abn(x, self.weight, self.bias, self.running_mean,
                           self.running_var, self.training, self.momentum,
                           self.eps, self.activation, self.slope)


class InPlaceABNSync(ABN):
    """InPlace Activated Batch Normalization with cross-GPU synchronization
    This assumes that it will be replicated across GPUs using the same
     mechanism as in `nn.DistributedDataParallel`.
    """

    def forward(self, x):
        return inplace_abn_sync(x, self.weight, self.bias, self.running_mean,
                                self.running_var, self.training, self.momentum,
                                self.eps, self.activation, self.slope)

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)
