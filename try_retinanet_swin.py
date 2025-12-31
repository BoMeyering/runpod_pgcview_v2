import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork


class SwinTFeatureExtractor(nn.Module):
    def __init__(self, weights=Swin_T_Weights.IMAGENET1K_V1, trainable=True):
        super().__init__()
        m = swin_t(weights=weights)

        self.body = create_feature_extractor(
            m,
            return_nodes={
                "features.1": "c2",
                "features.3": "c3",
                "features.5": "c4",
                "features.7": "c5",
            }
        )

    def forward(self, x):
        feats = self.body(x)  # dict of BHWC tensors
        out = OrderedDict()

        # BHWC -> NCHW for downstream FPN/detectors
        for k in ("c2", "c3", "c4", "c5"):
            v = feats[k]
            out[k] = v.permute(0, 3, 1, 2).contiguous()
        return out

class SwinFPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.backbone = SwinTFeatureExtractor()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[96, 192, 384, 768],
            out_channels=out_channels,
        )
        self.out_channels = out_channels

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.fpn(feats)

        return feats

def retinanet_swin(num_classes=2, backbone_out_channels=256):
    backbone = SwinFPN(out_channels=backbone_out_channels)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),   # one per level
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    model = RetinaNet(
        backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
    )

    model.detections_per_img = 100

    return model
    

model = retinanet_swin(num_classes=2)
model.eval()
model.score_thresh = 0.01

import cv2

img = cv2.imread('test_img.jpg', cv2.IMREAD_COLOR_RGB)
img = cv2.resize(img, (1024, 1024))  # Resize to
img_tensor = torch.from_numpy(img)  # Convert to tensor and normalize
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # Change to C,H,W and add batch dimension
print(img_tensor.shape)  # Should be (1, 3, H, W)


out = model(img_tensor)
print(out)

for _ in range(100):
    input = torch.randn(1, 3, 1024, 1024)
    output = model(input)
    print(output)


