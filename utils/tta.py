from collections import OrderedDict
from typing import Callable, Dict, List
from PIL import Image
import torchvision.transforms.functional as TF
import torch


def get_pil_tta_transforms() -> Dict[str, Callable[[Image.Image], Image.Image]]:
    """PIL-based TTA transforms used in the app UI.
    Returns an ordered mapping from augmentation name to a transform function.
    """
    return OrderedDict({
        'original': lambda img: img,
        'hflip': lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        'vflip': lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
        'rot90': lambda img: img.rotate(90, expand=True),
        'rot180': lambda img: img.rotate(180, expand=True),
        'rot270': lambda img: img.rotate(270, expand=True),
    })


def get_basic_tensor_tta_transforms() -> List[Callable[[torch.Tensor], torch.Tensor]]:
    """Tensor-based TTA transforms suitable for batched evaluation.
    Returns a list of callables operating on a CHW tensor.
    Includes identity and horizontal flip.
    """
    return [
        lambda t: t,  # original
        TF.hflip,
    ]
