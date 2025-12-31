import base64
import numpy as np
import cv2  
import torch
import onnxruntime as ort
import effdet
import segmentation_models_pytorch as smp
import json

img = cv2.imread('test_img.jpg', cv2.IMREAD_COLOR_RGB)
img = cv2.resize(img, (1024, 1024)) # Example image read
img_tensor = torch.from_numpy(img)  # Convert to tensor and normalize
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # Change to C,H,W and add batch dimension
print(img_tensor.shape)  # Should be (1, 3, H, W)

buffer = cv2.imencode('.png', img)[1]
print(buffer)

encoded = base64.b64encode(buffer).decode('utf-8') 

with open('test_input.json', 'w') as f:
    json.dump({
        'input': {'image': encoded}
    }, 
    f)


seg_model = smp.Unet(classes=4).eval()
effdet_config = effdet.get_efficientdet_config('tf_efficientdet_d1')
effdet_config.num_classes = 2
effdet_config.image_size = (1024, 1024)
effdet.config
marker_model = effdet.create_model_from_config(effdet_config, bench_task="predict").eval()

seg_model(img_tensor)
marker_model(img_tensor)

torch.onnx.export(seg_model, (img_tensor,), 'seg_model.onnx', input_names=['input'])
torch.onnx.export(marker_model, (img_tensor,), 'marker_model.onnx', input_names=['input'])

sess = ort.InferenceSession(
    'marker_model.onnx',
    providers=["CPUExecutionProvider"]
)

# out = sess.run(None, {'input': img_tensor.numpy()})

# print(out[0].shape)