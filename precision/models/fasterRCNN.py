import cv2
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
import torchvision.models.detection as models

from models.model import PyTorchModel

class FasterRCNNPytorchModel(PyTorchModel):
    fastrcnn_to_coco_id = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 
        10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 
        19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 
        28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 
        37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 
        46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 
        55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 
        64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 
        73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
    }

    def __init__(self, name: str, input_shape: tuple = (640, 640), device: str = None):
        super().__init__(name, input_shape, device)

        print(f"Loading {self.name} on {self.device}...")

        self.model = models.fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image):
        """
        Prétraitement Fast R-CNN (Standard ImageNet).
        Source: [2, 5]
        """
        self.original_h, self.original_w = image.shape[:2]
        target_h, target_w = self.input_shapes[0] # ex: 800, 800
        
        # Redimensionnement simple (pas de letterbox par défaut pour torchvision ONNX statique)
        # Note: Si le modèle supporte le dynamique, on pourrait garder le ratio.
        # Ici on assume un resize direct vers la taille d'entrée fixe du graphe.
        """ image_resized = cv2.resize(image, (target_w, target_h))
        
        # Conversion BGR -> RGB
        # image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalisation Statistique ImageNet
        image_float = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        image_normalized = (image_float - mean) / std
        
        # HWC -> CHW -> NCHW
        input_tensor = image_float.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0) """

        img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = torch.from_numpy(img).to(self.device)
        img = img.permute(2, 0, 1)
        img /= 255.0
        img = img.unsqueeze(0)
        # cv2.imshow("Preprocessed Image", img.cpu().numpy()[0].transpose(1, 2, 0))
        # cv2.waitKey(0)
        return img

    def draw_boxes(self, frame, boxes, scores, classes, image_id):
        for (box, score, cls_id) in zip(boxes, scores, classes):
            # Reverse letterbox transformation
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[0] + box[2])
            y2 = int(box[1] + box[3])

            label = f"{self.classes[int(cls_id)]} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(str(image_id), frame)
            cv2.waitKey(0)

    def postprocess(self, outputs, image,  image_id, conf_thres=0.0001, iou_thres=0.45):
        """
        Décodage Multi-Outputs (boxes, labels, scores).
        Les boîtes sortent généralement en [x1, y1, x2, y2] absolus.
        Source: [2, 23]
        """
        # On suppose l'ordre standard : boxes, labels, scores
        # Une implémentation robuste vérifierait les noms des outputs dans self.session
        pred = outputs[0]
        pred_boxes = pred["boxes"].cpu().numpy() # [N, 4]
        pred_labels = pred["labels"].cpu().numpy() # [N]
        pred_scores = pred["scores"].cpu().numpy() # [N]

        target_h, target_w = self.input_shapes[0]
        # Facteurs d'échelle pour revenir à la taille originale
        scale_x = self.original_w / target_w
        scale_y = self.original_h / target_h

        mask = pred_scores >= conf_thres
        boxes = pred_boxes[mask]
        scores = pred_scores[mask]
        class_ids = pred_labels[mask]

        # print(boxes)
        # print(scale_x, scale_y)
        
        # Redimensionnement vers l'image originale
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y

        # print(boxes)
        
        # Conversion [x1, y1, x2, y2] -> [x, y, w, h] pour COCO
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        indices = self._nms(boxes, scores, iou_threshold=0.5)

        final_boxes = boxes[indices]
        final_scores = scores[indices]
        final_class_ids = class_ids[indices]

        # self.draw_boxes(image, final_boxes, final_scores, final_class_ids, image_id)
            
        # Pas de NMS ici car souvent inclus dans le modèle Faster R-CNN exporté
        return self._format_coco(image_id, final_boxes, final_scores, final_class_ids)