"""
src/models.py
Load serialized models from '/models'
BoMeyering 2025
"""

import torch
import os
from typing import Union
from pathlib import Path


def load_model(file_path: Union[str, Path], device: Union[str, torch.device]):
    """
    Loads a Pytorch .pth as a model

    Parameters:
    -----------
        file_path : str, Path
            The relative file path to the model as a .pth or .pt filetype.
        device : str, torch.device
            The computational device the model should be mapped to.

    Returns:
    --------


    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The specified model path {file_path} does not exist."\
                "Please ensure that the correct path was specified."
            )
        elif not str(file_path).endswith(('.pt', '.pth')):
            raise ValueError(
                f"The specificd model path should be a 'pt' or '.pth' file type."
            )
        model = torch.load(file_path, map_location=device, weights_only=False)
        model.eval()

        return {
            'model': model,
        }
    except Exception as e:
        error_msg = f"Server encountered error loading the model at {file_path}: {str(e)}"

        return {
            'errors': error_msg
        }