# from ultralytics import YOLO
#
# model = YOLO("github-psat/rpi4/yolo5/yolov5nu.pt")
# model.info(detailed=True) # Prints GFLOPs, parameters, and layer info

# pip install thop
# import torch
# from thop import profile
# import torchvision.models.detection as models
#
# # model = models.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
# model = models.ssdlite320_mobilenet_v3_large(pretrained=True)
# input = torch.randn(1, 3, 640, 640) # YOLO/Faster R-CNN input size
# macs, params = profile(model, inputs=(input, ))
#
# print(f"Total FLOPs: {2 * macs}") # 1 MAC (Multiply-Accumulate) ≈ 2 FLOPs
# print(f"total GFLOPs: {2 * macs / 1e9}")

# pip install onnx-tool
import onnx_tool

# modelpath = 'github-psat/rpi4/yolo5/yolov5n.onnx'
# modelpath = 'github-psat/rpi4/yolo11/yolo11n-416.onnx'
modelpath = 'github-psat/rpi4/ssd/ssdlite_mobilenet_v3_large-320.onnx'
# model_profile will perform shape inference and count MACs/FLOPs
# onnx_tool.model_profile(modelpath)
onnx_tool.model_profile(modelpath, mcfg={'constant_folding': False})
