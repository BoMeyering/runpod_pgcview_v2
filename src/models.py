"""
src/models.py
Load serialized models from '/models'
BoMeyering 2025
"""

import torch
import os
import onnx
import onnxruntime as ort
from typing import Union, Dict
from pathlib import Path


def load_model(file_path: Union[str, Path], device: Union[str, torch.device]) -> Dict[str, Union[torch.nn.Module, str, None]]:
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
            'errors': None
        }
    except Exception as e:
        error_msg = f"Server encountered error loading the model at {file_path}: {str(e)}"

        return {
            'model': None,
            'errors': error_msg
        }
    

def load_onnx_model(onnx_path: Union[str, Path]) -> Union[ort.InferenceSession, Dict[str, str | None]]:
    """
    Loads an ONNX model from a specified file path.

    Parameters:
    -----------
        onnx_path : str, Path
            The relative file path to the ONNX model.

    Returns:
    --------
        Dict[str, Union[object, str, None]]
            An ONNX inference session set up for inference on CUDA first, then CPU.
    """

    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        for p in providers:
            if p not in ort.get_available_providers():
                raise ValueError(
                    f"Provider {p} not available in {ort.get_available_providers()}"\
                    "Please ensure that right providers are available in the runtime environment."
                )
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"The specified model path {onnx_path} does not exist."\
                "Please ensure that the correct path was specified."
            )
        
        sess = ort.InferenceSession(
            onnx_path,
            providers=providers
        )

        return sess
        
    except Exception as e:
        error_msg = f"Server encountered error loading the model at {onnx_path}: {str(e)}"

        return {
            'model': None,
            'errors': error_msg
        }
    