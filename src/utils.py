"""
src/utils.py
Handler utility functions
BoMeyering 2025
"""

import os
import io
import cv2
import base64
import numpy
import traceback
import numpy as np
from PIL import Image


def decode_img(b64_arr: str, expected_size: int=3145728) -> dict:
    """ Decode a base64 encoded PNG array """

    try:
        arr_bytes = base64.b64decode(b64_arr)                   # Decode base64 to get bytes
        png_encoded = np.frombuffer(arr_bytes, dtype=np.uint8)  # Load bytes into PNG array
        img = cv2.imdecode(png_encoded, cv2.IMREAD_COLOR)       # Decode PNG into numpy array

        if img.size > expected_size or img.size < expected_size: # Check if image size is more than (1024x1024x3)
            img = cv2.resize(img, (1024, 1024))
            
        # Return img_dict
        return {
            'image': img,
            'errors': None
        }
    except Exception as e:
        error_msg = f"Server encountered an error decoding the base64 image in 'decode_img()': {str(e)}"
        # Return error dict
        return {
            'image': None,
            'errors': error_msg
        }

def encode_out_map(out_map: numpy.ndarray) -> dict:
    """ Encode an np.uint8 output map in Base64 """

    try:
        _, buffer = cv2.imencode('.png', out_map)               # Encode output map as PNG
        encoded = base64.b64encode(buffer).decode('utf-8')      # Encode as base64 string

        return {
            'out_map': encoded,
            'errors': None
        }
    except Exception as e:
        error_msg = f"Server encountered an error encoding the segmentation output map in 'encode_out_map()': {str(e)}"
        return {
            'out_map': None,
            'errors': error_msg
        }

def encode_bbox_arr(bbox_arr: numpy.ndarray) -> dict:
    """ Encode the bounding box array as base64 """
    try:
        bbox_arr = bbox_arr.astype(np.float32)  # Cast to float32
        encoded = base64.b64encode(bbox_arr.tobytes()).decode('utf-8')  # Encode as base64

        return {
            'bboxes': encoded,
            'dtype': str(bbox_arr.dtype),
            'shape': bbox_arr.shape,
            'errors': None
        }
    except Exception as e:
        error_msg = f"Server encountered an error encoding the bbox array in 'encode_bbox_arr()': {str(e)}"

        return {
            'bboxes': None,
            'dtype': None,
            'shape': None,
            'errors': error_msg
        }


