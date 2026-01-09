# from ultralytics import YOLO
#
# model = YOLO("github-psat/rpi4/yolo5/yolov5nu.pt")
# model.info(detailed=True) # Prints GFLOPs, parameters, and layer info

# pip install thop
import torch
from thop import profile
import torchvision.models.detection as models

# model = models.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model = models.ssdlite320_mobilenet_v3_large(pretrained=True)
input = torch.randn(1, 3, 640, 640) # YOLO/Faster R-CNN input size
macs, params = profile(model, inputs=(input, ))

print(f"Total FLOPs: {2 * macs}") # 1 MAC (Multiply-Accumulate) ≈ 2 FLOPs
print(f"total GFLOPs: {2 * macs / 1e9}")
