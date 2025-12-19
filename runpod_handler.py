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
import torch.functional as F
from dotenv import load_dotenv
from pathlib import Path
from torchvision.transforms import ToTensor
from src.models import load_model
from src.utils import decode_img, encode_bbox_arr, encode_out_map
from src.transforms import get_inference_transforms

#-------------------------------------#
# Server Initiatlization
#-------------------------------------#
# Get model paths
load_dotenv()
seg_model_path = os.getenv('SEG_MODEL_PATH')
marker_model_path = os.getenv('MARKER_MODEL_PATH')

if marker_model_path is None or seg_model_path is None:
    sys.exit("Model paths were not defined in the ENV")

# Set the torch device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Set transforms
transforms = get_inference_transforms()

# Load models
try:
    marker_model = load_model(file_path=marker_model_path, device=device)['model']
    seg_model = load_model(file_path=seg_model_path, device=device)['model']

    marker_model = marker_model.eval().to(device)
    seg_model = seg_model.eval().to(device)
except Exception as e:
    sys.exit(f"Call to load_model() failed: {str(e)}")

#-------------------------------------#
# Server API Request Handler
#-------------------------------------#
def handler(request):
    """ Handles API requests and processes them """

    print(f"[INFO] Received request at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()

    input_data = request.get('input', {})
    b64_img = input_data.get('image')

    # Return error if image is missing
    if b64_img is None:
        return {
            "errors": "Missing 'image' key in request payload."
        }

    # Decode image and collect errors
    img_dict = decode_img(b64_img) # Returns a Numpy array
    if img_dict.get('errors') is not None:
        return {
            'errors': img_dict.get('errors')
        }

    # Create normalized tensors for each model
    img = img_dict.get('image')
    effdet_tensor = effdet_transforms(image=img)['image'].unsqueeze(0).to(device)
    dlv3p_tensor = dlv3p_transforms(image=img)['image'].unsqueeze(0).to(device)

    # Forward pass for effdet model
    try:
        with torch.no_grad():
            effdet_out = effdet_model(effdet_tensor).squeeze().detach().cpu().numpy()
            effdet_dict = encode_bbox_arr(effdet_out)
            
            # Return errors associated with 
            if effdet_dict.get('errors') is not None:
                return {
                    'errors': effdet_dict.get('errors')
                }
    except Exception as e:
        error_msg = f"Server encountered error processing the image through the EffDet model: {str(e)}"
        return {
            'errors': error_msg
        }

    # Forward pass for dlv3p model
    try:
        with torch.no_grad():
            dlv3p_out = dlv3p_model(dlv3p_tensor)
            dlv3p_out = torch.argmax(dlv3p_out, dim=1).squeeze(0)
            dlv3p_out = dlv3p_out.detach().cpu().numpy().astype(np.uint8)
            dlv3p_dict = encode_out_map(dlv3p_out)

            if dlv3p_dict.get("errors") is not None:
                return {
                    "errors": dlv3p_dict.get("errors")
                }
    except Exception as e:
        error_msg = f"Server encountered error processing the image through the DeepLabV3Plus model: {str(e)}"
        return {
            "errors": error_msg
        }

    p_time = time.time() - t0
    

    return {
        'effdet': effdet_dict,
        'dlv3p': dlv3p_dict,
        'server_inference_time': p_time
    }
#-------------------------------------#
# Main Function
#-------------------------------------#
# Start the Serverless function when the container runs the script
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})