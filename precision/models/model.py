import numpy as np
import onnxruntime as ort
import torch
import cv2
from ultralytics import YOLO

class Model:
    """Classe de base abstraite pour l'unification des interfaces d'inférence."""
    
    def __init__(self, name: str):
        self.name = name

    def preprocess(self, image: np.array):
        raise NotImplementedError("La méthode preprocess doit être implémentée par la sous-classe.")

    def run(self, image: np.array, image_id: int):
        raise NotImplementedError("La méthode run doit être implémentée par la sous-classe.")

    def postprocess(self, outputs: np.array, image: np.array, image_id: int):
        raise NotImplementedError("La méthode postprocess doit être implémentée par la sous-classe.")
    
    def _format_coco(self, image_id, boxes, scores, class_ids, mapping: dict=None):
        """Helper pour formater la sortie selon le standard COCO."""
        detections = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            if mapping:
                category_id = mapping[int(cls_id)]
            else:
                category_id = int(cls_id)

            detections.append({
                "image_id": image_id,
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])], # x, y, w, h
                "score": float(score),
                "label": self.classes[category_id],
                "category_id": category_id
            })
        return detections

    def _nms(self, boxes, scores, iou_threshold=0.45):
        """
        Applies Non-Maximum Suppression (NMS).
        
        Args:
            boxes: np.array of shape (N, 4)
            scores: np.array of shape (N,)
            iou_threshold: float
        """
        # 1. Flatten inputs to ensure they are 1D/2D (Handle (1, N, 4) or (N, 80) cases)
        if boxes.ndim == 3: boxes = boxes.squeeze(0)  # Remove batch dim if present
        if scores.ndim > 1: scores = scores.flatten() # Flatten scores if (1, N, C)
        
        if len(boxes) == 0: return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2] # Be careful: Is input [x,y,w,h] or [x1,y1,x2,y2]? 
        y2 = boxes[:, 1] + boxes[:, 3] # Adjust this math based on your input format
        
        areas = (x2 - x1) * (y2 - y1) # Recalculate area based on coords
        order = scores.argsort()[::-1]

        keep = []
        
        while order.size > 0:
            i = order[0] 
            keep.append(i)
            
            # Compare best box (i) with all remaining boxes (order[1:])
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0] 
            
            # Update order to keep only boxes with low IoU
            order = order[inds + 1]

        return np.array(keep)

class ONNXModel(Model):
    """Classe de base abstraite pour l'unification des interfaces d'inférence."""

    classes = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
        67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
        75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
        86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }
    
    def __init__(self, path: str, name: str, provider: str='CPUExecutionProvider'):
        super().__init__(name)
        self.session = ort.InferenceSession(path, providers=[provider])
        self.input_names = [input.name for input in self.session.get_inputs()]
        # On suppose une forme statique pour simplifier, 
        # mais le code pourrait lire dynamiquement.
        self.input_shapes = [input.shape[2:] for input in self.session.get_inputs()]
        self.type = 'onnx'

    def preprocess(self, image: np.array):
        raise NotImplementedError("La méthode preprocess doit être implémentée par la sous-classe.")

    def run(self, image: np.array, image_id: int):
        # Exécution standard ONNX
        input_tensor = self.preprocess(image)
        result = self.session.run(None, {self.input_names[0]: input_tensor})
        return self.postprocess(result, image, image_id)

    def postprocess(self, outputs: np.array, image: np.array, image_id: int):
        raise NotImplementedError("La méthode postprocess doit être implémentée par la sous-classe.")
    
class PyTorchModel(Model):
    """
    Abstract base class for PyTorch inference interfaces.
    Maintains compatibility with previous NumPy-based post-processing.
    """

    # COCO Class Mapping (Standard 91-class mapping often used in Torchvision)
    # Note: If using YOLO, you might need to map these to 0-79.
    classes = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
        67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
        75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
        86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }

    def __init__(self, name: str, input_shape: tuple = (640, 640), device: str = None):
        """
        Args:
            model: Can be a path (str) to a .pt file or a loaded torch.nn.Module object.
            name: Name of the model (for display/logging).
            input_shape: tuple (H, W) expected by the model (default 640x640).
            device: 'cuda', 'cpu', or 'mps'. If None, auto-detects.
        """
        super().__init__(name)
        self.input_shapes = [input_shape] # Stored as list to match previous structure

        # 1. Device Setup
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps') # For Mac M1/M2/M3
            else:
                self.device = torch.device('cpu')

        self.type = 'pytorch'
    

    def preprocess(self, image: np.array):
        raise NotImplementedError("Subclass must implement preprocess.")

    def run(self, image: np.array, image_id: int):
        # 1. Preprocess (returns typically a numpy array)
        input_data = self.preprocess(image)

        # 2. Convert to Tensor
        if isinstance(input_data, np.ndarray):
            # PyTorch expects (Batch, Channel, Height, Width)
            # Ensure input is 4D. If 3D (C, H, W), add batch dim.
            if input_data.ndim == 3:
                input_data = np.expand_dims(input_data, axis=0)
            
            # Create tensor, ensure float32, and move to device
            input_tensor = torch.from_numpy(input_data).float().to(self.device)
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data.to(self.device)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        # 3. Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # 4. Convert outputs back to NumPy/CPU for the existing postprocess logic
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        elif isinstance(outputs, (list, tuple)):
            # Handle cases like Faster R-CNN which return list of dicts or tuples
            outputs = [
                x.cpu().numpy() if isinstance(x, torch.Tensor) else x 
                for x in outputs
            ]
        elif isinstance(outputs, dict):
            # Handle dictionary outputs
            outputs = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                for k, v in outputs.items()
            }

        return self.postprocess(outputs, image, image_id)

    def postprocess(self, outputs, image, image_id: int):
        raise NotImplementedError("Subclass must implement postprocess.")

class UltralyticsModel(Model):
    """
    Abstract base class for PyTorch inference interfaces.
    Maintains compatibility with previous NumPy-based post-processing.
    """

    # COCO Class Mapping (Standard 91-class mapping often used in Torchvision)
    # Note: If using YOLO, you might need to map these to 0-79.
    classes = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
        67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
        75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
        80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
        86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }

    def __init__(self, name: str, path: str, input_shape: tuple = (640, 640)):
        """
        Args:
            model: Can be a path (str) to a .pt file or a loaded torch.nn.Module object.
            name: Name of the model (for display/logging).
            input_shape: tuple (H, W) expected by the model (default 640x640).
            device: 'cuda', 'cpu', or 'mps'. If None, auto-detects.
        """
        super().__init__(name)
        print(f"Loading {self.name}...")

        self.model = YOLO(path)
        self.input_shapes = [input_shape] # Stored as list to match previous structure

        self.type = 'pytorch'
    

    def preprocess(self, image: np.array):
        raise NotImplementedError("Subclass must implement preprocess.")

    def run(self, image: np.array, image_id: int):
        # 1. Preprocess (returns typically a numpy array)
        input_data = self.preprocess(image)

        outputs = self.model.predict(
            source=input_data, 
            imgsz=self.input_shapes[0], 
            conf=0.0001,
            iou=0.45,
            verbose=False
        )

        return self.postprocess(outputs, image, image_id)

    def postprocess(self, outputs, image, image_id: int):
        raise NotImplementedError("Subclass must implement postprocess.")
