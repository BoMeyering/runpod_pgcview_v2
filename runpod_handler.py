"""
runpod_handler.py
RunPod.io serverless handler script
BoMeyering 2025
"""

import runpod
import time
import torch
import os
import sys
import numpy as np
import onnxruntime as ort
import torch.functional as F
from dotenv import load_dotenv
from pathlib import Path
from torchvision.transforms import ToTensor
from src.models import load_model, load_onnx_model
from src.utils import decode_img, encode_bbox_arr, encode_out_map
from src.transforms import get_inference_transforms

#-------------------------------------#
# Server Initiatlization
#-------------------------------------#
# Get model paths
load_dotenv()
seg_model_path = os.getenv('SEG_ONNX_PATH')
marker_model_path = os.getenv('MARKER_ONNX_PATH')

if marker_model_path is None or seg_model_path is None:
    sys.exit("Model paths were not defined in the ENV")

# Set the torch device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Set transforms
transforms = get_inference_transforms()

# Load models
try:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # for p in providers:
    #     if p not in ort.get_available_providers():
    #         raise ValueError(
    #             f"Provider {p} not available in {ort.get_available_providers()}"\
    #             "Please ensure that right providers are available in the runtime environment."
    #         )
    if not os.path.exists(marker_model_path):
        raise FileNotFoundError(
            f"The specified model path {marker_model_path} does not exist."\
            "Please ensure that the correct path was specified."
        )
    if not os.path.exists(seg_model_path):
        raise FileNotFoundError(
            f"The specified model path {seg_model_path} does not exist."\
            "Please ensure that the correct path was specified."
        )
    
    # Create an ONNX Inference Session for the marker model
    marker_onnx_sess = ort.InferenceSession(
        marker_model_path,
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 4 * 1024**3,  # 4GB cap (optional)
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    )

    # Create an ONNX Inference Session for the segmentation model
    seg_onnx_sess = ort.InferenceSession(
        seg_model_path,
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 4 * 1024**3,  # 4GB cap (optional)
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    )

except Exception as e:
    sys.exit(f"Failed to load ONNX models and create sessions: {str(e)}")

#-------------------------------------#
# Server API Request Handler
#-------------------------------------#
def handler(request):
    """ 
    Handles and API request with data and processes it 

    Parameters:
    -----------
    request : JSON or dict like
        A dictionary object with at minimum the following structure
        {
            "inputs": {
                "image": <BASE64 encoded str>
            }
        }

    Returns:
    --------
    
    """

    print(f"[INFO] Received request at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()

    input_data = request.get('input', None)
    if input_data is None:
        return {
            "errors": "Missing 'input' key in request paylod"
        }

    b64_img = input_data.get('image', None)
    if b64_img is None:
        return {
            "errors": "Missing 'image' key in request input payload."
        }

    # Decode image and collect errors
    img_dict = decode_img(b64_img) # Returns a Numpy array
    if img_dict.get('errors') is not None:
        return {
            'errors': img_dict.get('errors')
        }

    # Create normalized tensors for each model
    img = img_dict.get('image')
    img_array = transforms(image=img)['image']
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)  # Change to C,H,W and add batch dimension

    # Forward pass for effdet model
    try:
        marker_out = marker_onnx_sess.run(
            None,
            {'input': img_array}
        )

        marker_out = marker_out[0].squeeze()

        print(marker_out)
        print(marker_out.shape)
        
        marker_dict = encode_bbox_arr(marker_out)
            
        # Return errors associated with marker_dict
        if marker_dict.get('errors') is not None:
            return {
                'errors': marker_dict.get('errors')
            }
    except Exception as e:
        error_msg = f"Server encountered error processing the image through the EffDet model: {str(e)}"
        return {
            'errors': error_msg
        }

    # Forward pass for dlv3p model
    try:
        seg_out = seg_onnx_sess.run(
            None, 
            {'input': img_array}
        )

        seg_out = seg_out[0].squeeze()
        seg_out = np.argmax(seg_out, axis=1).astype(np.uint8)   # Find the best prediction
        seg_dict = encode_out_map(seg_out)                      # Encode in base64

        if seg_dict.get("errors") is not None:
            return {
                "errors": seg_dict.get("errors")
            }
    except Exception as e:
        error_msg = f"Server encountered error processing the image through the segmentation model: {str(e)}"
        return {
            "errors": error_msg
        }

    p_time = time.time() - t0
    
    return {
        'marker': marker_dict,
        'segmentation': seg_dict,
        'server_inference_time': p_time
    }

#-------------------------------------#
# Main Function
#-------------------------------------#
# Start the Serverless function when the container runs the script
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})