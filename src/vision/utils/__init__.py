from typing import Any, Callable, Dict, Mapping, Sequence, Type, Union

from torch import Tensor
from torch.hub import load_state_dict_from_url

from src.vision.utils.dependency import check_version
from src.vision.utils.hooks import FeatureExtractor
from src.vision.utils.image_utils import (
    cv2_imshow,
    get_image_from_url,
    read_image_to_tensor,
)
from src.vision.utils.visualizer import Visualizer

__all__ = [
    "check_version",
    "contains_any_tensor",
    "cv2_imshow",
    "get_image_from_url",
    "get_callable_dict",
    "load_state_dict_from_url",
    "read_image_to_tensor",
    "FeatureExtractor",
    "Visualizer",
]


def get_callable_name(fn_or_class: Union[Callable, object]) -> str:
    return getattr(fn_or_class, "__name__", fn_or_class.__class__.__name__).lower()


def get_callable_dict(fn: Union[Callable, Mapping, Sequence]) -> Union[Dict, Mapping]:
    if isinstance(fn, Mapping):
        return fn
    elif isinstance(fn, Sequence):
        return {get_callable_name(f): f for f in fn}
    elif callable(fn):
        return {get_callable_name(fn): fn}


def contains_any_tensor(value: Any, dtype: Type = Tensor) -> bool:
    """Determine whether or not a list contains any Type."""
    if isinstance(value, dtype):
        return True
    if isinstance(value, (list, tuple)):
        return any(contains_any_tensor(v, dtype=dtype) for v in value)
    elif isinstance(value, dict):
        return any(contains_any_tensor(v, dtype=dtype) for v in value.values())
    return False
