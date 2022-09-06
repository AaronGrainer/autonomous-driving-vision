from functools import wraps
from typing import Tuple

from torch.utils.data.dataset import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Subclass of torch.utils.data.dataset.Dataset that enables on the fly resizing
    of `input_dim`

    Args:
        input_dimension (tuple): (width, height) tuple with default network dimension
        mosiac (bool): Determine if data should be mosiac augmented
    """

    def __init__(self, input_dimension: Tuple, mosiac=True):
        super().__init__()
        self.__input_dim = input_dimension[:2]
        self.enable_mosaic = mosiac

    @property
    def input_dim(self):
        """Dimension that can be used by transformers to set the correct image size, etc.

        This allows the transformers to have a single source of truth for the input
        dimension of the network.

        Return:
            list: Tuple containing the current (weight, height)
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim
        return self.__input_dim

    @staticmethod
    def mosaic_getitem(getitem_fn):
        """Decorator method that needs to be used around the `__getitem__ method.

        This decorator enables the closing mosaic

        Example:
            >>> class CustomSet(ln.data.Dataset):
            ...     def __len__(self):
            ...         return 10
            ...
            ...     @ln.data.Dataset.mosaic_getitem
            ...     def __getitem__(self, index):
            ...         return self.enable_mosaic
        """

        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                self.enable_mosaic = index[0]
                index = index[1]

            ret_val = getitem_fn(self, index)

            return ret_val

        return wrapper
