"""
src/models.py
Load serialized models from '/models'
BoMeyering 2025
"""

import torch
import os
# import onnx
# import onnxruntime as ort
import torch.nn as nn
from typing import Union, Dict
from pathlib import Path
import omegaconf
from omegaconf import OmegaConf
from enum import Enum
import segmentation_models_pytorch as smp
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import create_model
from effdet.bench import DetBenchTrain, DetBenchPredict
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet

class ModelArchitecture(Enum):
    """
    Enumeration of valid segmentation model architectures from segmentation_models_pytorch.
    """
    
    DEEPLABV3 = 'DeepLabV3'
    DEEPLABV3PLUS = 'DeepLabV3Plus'
    FPN = 'FPN'
    LINKNET = 'Linknet'
    MANET = 'MAnet'
    PAN = 'PAN'
    PSPNET = 'PSPNet'
    SEGFORMER = 'Segformer'
    UPERNET = 'UPerNet'
    UNET = 'Unet'
    UNETPLUSPLUS = 'UnetPlusPlus'

def create_smp_model(conf: omegaconf.dictconfig.DictConfig) -> torch.nn.Module:
    """Creates an smp Pytorch model

    conf:
        conf (omegaconf.dictconfig.DictConfig): The OmegaConf configuration dictionary

    Raises:
        ValueError: If conf.model.config.encoder_name is not listed in smp.encoders.get_encoder_names().
        ValueError: If conf.model.architecture does not match any of the specified architectures.

    Returns:
        torch.nn.Module: A model as a pytorch module
    """

    # Select the model cofiguration
    try:
        model_config = conf.model.config
    except Exception as e:
        raise Exception(e)
    
    if model_config.encoder_name not in smp.encoders.get_encoder_names():
        raise ValueError(f"Encoder name {model_config.encoder_name} is not one of the accepted encoders. Please select an encoder from {smp.encoders.get_encoder_names()}")
    
    try:
        model_class = getattr(smp, conf.model.architecture.value)
        model = model_class(**model_config)

        return model
    except AttributeError as e:
        raise ValueError(f"Model architecture {conf.model.architecture} is not a valid SMP architecture.\nSelect one from 'smp._MODEL_ARCHITECTURES'")

class EffDetWrapper(nn.Module):
    def __init__(self, conf: OmegaConf, device: torch.device):
        super().__init__()
        try:
            self.model = create_model(
                model_name=conf.effdet.architecture,
                pretrained=True,
                num_classes=conf.model.num_classes,
                image_size=(conf.images.resize, conf.images.resize),
                max_det_per_image=conf.model.detections_per_img,
            ).to(device)
        except Exception as e:
            print(f"Error creating model: {e}")
            raise

        # Keep an explicit config for benches
        self.config = get_efficientdet_config(conf.effdet.architecture)
        self.config.num_classes = conf.model.num_classes
        self.config.image_size = conf.images.resize

        self.train_bench = DetBenchTrain(self.model).to(device)
        self.eval_bench  = DetBenchPredict(self.model).to(device)

    def train_mode(self):
        self.train_bench.train()
        self.eval_bench.train()

    def eval_mode(self):
        self.train_bench.eval()
        self.eval_bench.eval()

    def forward_train(self, images, targets):

        return self.train_bench(images, targets)
    
    @torch.no_grad()
    def predict(self, images):

        if images.ndim == 3:
            images = images.unsqueeze(0)

        return self.eval_bench(images)

def load_segformer_model(
        weights_path: Union[str, Path], 
        conf_path: Union[str, Path], 
        device: Union[str, torch.device]
    ) -> Dict[str, Union[torch.nn.Module, str, None]]:
    """
    Loads a Pytorch .pth as a model

    Parameters:
    -----------
        weights_path : str, Path
            The relative file path to the model weights as a .pth or .pt filetype.
        conf_path : str, Path
            The relative file path to the model configuration as a .yaml or .yml filetype.
        device : str, torch.device
            The computational device the model should be mapped to.

    Returns:
    --------


    """
    try:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"The specified model path {weights_path} does not exist."\
                "Please ensure that the correct path was specified."
            )
        elif not str(weights_path).endswith(('.pt', '.pth')):
            raise ValueError(
                f"The specificd model path should be a 'pt' or '.pth' file type."
            )
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        ema_weights = state_dict.get('ema_state_dict', None)
        if ema_weights is not None:
            print("EMA weights found in the checkpoint. Using EMA weights for model loading.")
            state_dict = ema_weights

        if not os.path.exists(conf_path):
            raise FileNotFoundError(
                f"The specified config path {conf_path} does not exist."\
                "Please ensure that the correct path was specified."
            )
        elif not str(conf_path).endswith(('.yaml', '.yml')):
            raise ValueError(
                f"The specified config path should be a 'yaml' or '.yml' file type."
            )
        conf = OmegaConf.load(conf_path)
        conf.model.architecture = ModelArchitecture[conf.model.architecture]

        model = create_smp_model(conf).to(device)
        model.load_state_dict(ema_weights)
        model.eval()

        return {
            'model': model,
            'errors': None
        }
    except Exception as e:
        error_msg = f"Server encountered error loading the model at {weights_path}: {str(e)}"

        return {
            'model': None,
            'errors': error_msg
        }
    
def load_effdet_model(
        weights_path: Union[str, Path], 
        conf_path: Union[str, Path], 
        device: torch.device
    )-> Dict[str, Union[EffDetWrapper, str, None]]:
    """_summary_

    Args:
        weights_path (Union[str, Path]): _description_
        conf_path (Union[str, Path]): _description_
        device (torch.device): _description_

    Returns:
        Dict[str, Union[EffDetWrapper, str, None]]: _description_
    """
    try:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"The specified model path {weights_path} does not exist."\
                "Please ensure that the correct path was specified."
            )
        elif not str(weights_path).endswith(('.pt', '.pth')):
            raise ValueError(
                f"The specified model path should be a 'pt' or '.pth' file type."
            )
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        
        if not os.path.exists(conf_path):
            raise FileNotFoundError(
                f"The specified config path {conf_path} does not exist."\
                "Please ensure that the correct path was specified."
            )
        elif not str(conf_path).endswith(('.yaml', '.yml')):
            raise ValueError(
                f"The specified config path should be a 'yaml' or '.yml' file type."
            )
        conf = OmegaConf.load(conf_path)
        model = EffDetWrapper(conf, device)
        model.load_state_dict(state_dict)
        model.eval_mode()
        return {
            'model': model,
            'errors': None
        }
    except Exception as e:
        error_msg = f"Server encountered error loading the model at {weights_path} with config {conf_path}: {str(e)}"

        return {
            'model': None,
            'errors': error_msg
        }


# def load_onnx_model(onnx_path: Union[str, Path]) -> Union[ort.InferenceSession, Dict[str, str | None]]:
#     """
#     Loads an ONNX model from a specified file path.

#     Parameters:
#     -----------
#         onnx_path : str, Path
#             The relative file path to the ONNX model.

#     Returns:
#     --------
#         Dict[str, Union[object, str, None]]
#             An ONNX inference session set up for inference on CUDA first, then CPU.
#     """

#     try:
#         providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#         for p in providers:
#             if p not in ort.get_available_providers():
#                 raise ValueError(
#                     f"Provider {p} not available in {ort.get_available_providers()}"\
#                     "Please ensure that right providers are available in the runtime environment."
#                 )
#         if not os.path.exists(onnx_path):
#             raise FileNotFoundError(
#                 f"The specified model path {onnx_path} does not exist."\
#                 "Please ensure that the correct path was specified."
#             )
        
#         sess = ort.InferenceSession(
#             onnx_path,
#             providers=providers
#         )

#         return sess
        
#     except Exception as e:
#         error_msg = f"Server encountered error loading the model at {onnx_path}: {str(e)}"

#         return {
#             'model': None,
#             'errors': error_msg
#         }

