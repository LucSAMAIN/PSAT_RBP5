python3 evaluate.py --model-name yolov5 --type onnx --model-path ../rpi4/yolo5/yolov5n.onnx --output ../data/outputs/yolov5_onnx.json
python3 evaluate.py --model-name yolov11 --type onnx --model-path ../rpi4/yolo11/yolo11n-416.onnx --output ../data/outputs/yolov11_onnx.json
python3 evaluate.py --model-name fasterRCNN --type pytorch --output ../data/outputs/fasterRCNN_pytorch.json
python3 evaluate.py --model-name ssd --type onnx --model-path ../rpi4/ssd/ssdlite_mobilenet_v3_large-320.onnx --output ../data/outputs/ssd_onnx.json
python3 evaluate.py --model-name yolov5 --type pytorch --model-path ../rpi4/yolo5/yolov5n.pt --output ../data/outputs/yolov5_pytorch.json
python3 evaluate.py --model-name yolov11 --type pytorch --model-path ../rpi4/yolo11/yolo11n.pt --output ../data/outputs/yolov11_pytorch.json
python3 evaluate.py --model-name ssd --type pytorch --model-path ../rpi4/ssd/ssdlite_mobilenet_v3_large-320.onnx --output ../data/outputs/ssd_pytorch.json