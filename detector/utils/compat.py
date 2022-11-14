import torch


_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]


def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)

