"""
src/transforms.py
All image transformations before modeling
BoMeyering 2025
"""

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_inference_transforms(
        means: Iterable=(0.485, 0.456, 0.406), 
        std: Iterable=(0.229, 0.224, 0.225)
    ) -> albumentations.Compose:
    """
    Composes an inference transform function that normalizes and converts to a tensor.

    Parameters:
    -----------
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to ImageNet means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to ImageNet std.

    Returns:
    --------
        albumentations.Compose: A Compose function to use in the datasets
    """

    transforms = A.Compose(
        [
            A.Normalize(mean=means, std=std, p=1.0),
            ToTensorV2(p=1.0)
        ]
    )
 
    return transforms