import cv2
import numpy as np
import torch

from models.model import ONNXModel, PyTorchModel, UltralyticsModel

class YoloV5ONNXModel(ONNXModel):
    yolo_to_coco_id = {
        0: 1,    1: 2,    2: 3,    3: 4,    4: 5,    5: 6,    6: 7,    7: 8,    8: 9,    9: 10,
        10: 11,  11: 13,  12: 14,  13: 15,  14: 16,  15: 17,  16: 18,  17: 19,  18: 20,  19: 21,
        20: 22,  21: 23,  22: 24,  23: 25,  24: 27,  25: 28,  26: 31,  27: 32,  28: 33,  29: 34,
        30: 35,  31: 36,  32: 37,  33: 38,  34: 39,  35: 40,  36: 41,  37: 42,  38: 43,  39: 44,
        40: 46,  41: 47,  42: 48,  43: 49,  44: 50,  45: 51,  46: 52,  47: 53,  48: 54,  49: 55,
        50: 56,  51: 57,  52: 58,  53: 59,  54: 60,  55: 61,  56: 62,  57: 63,  58: 64,  59: 65,
        60: 67,  61: 70,  62: 72,  63: 73,  64: 74,  65: 75,  66: 76,  67: 77,  68: 78,  69: 79,
        70: 80,  71: 81,  72: 82,  73: 84,  74: 85,  75: 86,  76: 87,  77: 88,  78: 89,  79: 90
    }
    
    def preprocess(self, image):
        """
        Implémente le 'Letterbox resizing' spécifique à YOLOv5.
        Source: 
        """
        self.original_h, self.original_w = image.shape[:2]
        target_h, target_w = self.input_shapes[0]
        
        # Calcul du ratio pour conserver l'aspect
        scale = min(target_w / self.original_w, target_h / self.original_h)
        nw, nh = int(self.original_w * scale), int(self.original_h * scale)
        
        # Redimensionnement
        image_resized = cv2.resize(image, (nw, nh))
        
        # Padding pour atteindre les dimensions cibles (souvent carré)
        # YOLOv5 utilise souvent une couleur de remplissage (114, 114, 114)
        dw, dh = target_w - nw, target_h - nh
        # Centrage du padding (divisé par 2)
        dw /= 2
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, 
                                          cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Sauvegarde des paramètres pour le post-traitement (inversion)
        self.pad_params = (scale, left, top)
        
        # Conversion BGR -> RGB et Normalisation 0-1
        # Layout: HWC -> CHW
        blob = cv2.dnn.blobFromImage(image_padded, 1/255.0, swapRB=True, crop=False)
        return blob

    def draw_boxes(self, frame, boxes, scores, classes, image_id):
        for (box, score, cls_id) in zip(boxes, scores, classes):
            # Reverse letterbox transformation
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[0] + box[2])
            y2 = int(box[1] + box[3])

            label = f"{self.classes[int(cls_id) - 1]} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(str(image_id), frame)
        cv2.waitKey(0)

    def postprocess(self, outputs, image, image_id, conf_thres=0.0001, iou_thres=0.45):
        """
        Optimized YOLOv5 post-processing.
        """
        # 1. Handle Batch Dimension (Assume batch_size=1 for simplicity)
        # output[0] shape: (1, 25200, 85) -> squeeze to (25200, 85)
        predictions = np.squeeze(outputs[0])

        # 2. Compute Scores and Class IDs (Vectorized)
        # objectness (N, 1) * class_probs (N, 80) -> scores (N, 80)
        objectness = predictions[:, 4]
        class_probs = predictions[:, 5:]
        
        # We can multiply (N,) * (N, 80) using broadcasting
        all_scores = class_probs * objectness[:, np.newaxis]

        # Find the best class and score for each box
        class_ids = np.argmax(all_scores, axis=1) # Shape (N,)
        scores = np.max(all_scores, axis=1)       # Shape (N,)

        # 3. FILTERING (Critical Performance Step)
        # Keep only boxes with score > conf_thres
        mask = scores > conf_thres
        
        if not np.any(mask):
            return []

        # Apply mask to data
        predictions = predictions[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # 4. Decode Coordinates (Only for kept boxes)
        # Extract box params: cx, cy, w, h
        cx = predictions[:, 0]
        cy = predictions[:, 1]
        w = predictions[:, 2]
        h = predictions[:, 3]

        # Convert to Top-Left (x, y)
        x = cx - (w / 2)
        y = cy - (h / 2)

        # 5. Invert Letterbox / Scaling
        scale, pad_left, pad_top = self.pad_params
        
        x = (x - pad_left) / scale
        y = (y - pad_top) / scale
        w = w / scale
        h = h / scale

        # Stack into (M, 4) array
        boxes = np.stack([x, y, w, h], axis=1)

        # 6. NMS
        # Ensure scores is 1D array of floats
        indices = self._nms(boxes, scores, iou_thres)
        if len(indices) == 0:
            return []

        # Select final results
        final_boxes = boxes[indices]
        final_scores = scores[indices]
        final_class_ids = class_ids[indices]
        #print(f"Final Boxes: {final_boxes}, Final Scores: {final_scores}, Final Class IDs: {final_class_ids}")
        # self.draw_boxes(image, final_boxes, final_scores, final_class_ids, image_id)

        return self._format_coco(image_id, final_boxes, final_scores, final_class_ids, self.yolo_to_coco_id)

class YoloV5UltralyticsModel(UltralyticsModel):

    yolo_to_coco_id = {
        0: 1,    1: 2,    2: 3,    3: 4,    4: 5,    5: 6,    6: 7,    7: 8,    8: 9,    9: 10,
        10: 11,  11: 13,  12: 14,  13: 15,  14: 16,  15: 17,  16: 18,  17: 19,  18: 20,  19: 21,
        20: 22,  21: 23,  22: 24,  23: 25,  24: 27,  25: 28,  26: 31,  27: 32,  28: 33,  29: 34,
        30: 35,  31: 36,  32: 37,  33: 38,  34: 39,  35: 40,  36: 41,  37: 42,  38: 43,  39: 44,
        40: 46,  41: 47,  42: 48,  43: 49,  44: 50,  45: 51,  46: 52,  47: 53,  48: 54,  49: 55,
        50: 56,  51: 57,  52: 58,  53: 59,  54: 60,  55: 61,  56: 62,  57: 63,  58: 64,  59: 65,
        60: 67,  61: 70,  62: 72,  63: 73,  64: 74,  65: 75,  66: 76,  67: 77,  68: 78,  69: 79,
        70: 80,  71: 81,  72: 82,  73: 84,  74: 85,  75: 86,  76: 87,  77: 88,  78: 89,  79: 90
    }

    def preprocess(self, image):
        """
        Implémente le 'Letterbox resizing' spécifique à YOLOv5.
        Source: 
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def draw_boxes(self, frame, boxes, scores, classes, image_id):
        for (box, score, cls_id) in zip(boxes, scores, classes):
            # Reverse letterbox transformation
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[0] + box[2])
            y2 = int(box[1] + box[3])

            label = f"{self.classes[self.yolo_to_coco_id[int(cls_id)]]} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(str(image_id), frame)
        cv2.waitKey(0)

    def postprocess(self, outputs, image, image_id, conf_thres=0.0001, iou_thres=0.45):
        """
        Optimized YOLOv5 post-processing.
        """
        if not outputs or not outputs[0].boxes:
            return self._format_coco(image_id, [], [], [], self.yolo_to_coco_id)

        # Extract all boxes data at once
        boxes_data = outputs[0].boxes
        
        # Extract components from the Boxes object
        # xyxy is normalized coordinates (x1, y1, x2, y2)
        # The .data tensor contains [x1, y1, x2, y2, confidence, class_id]
        detections = boxes_data.data.cpu().numpy()

        if len(detections) > 0:
            # Extract components
            boxes = detections[:, :4]
            scores = detections[:, 4]
            class_ids = detections[:, 5].astype(int)

            final_boxes = boxes
            final_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
            final_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height

        # self.draw_boxes(image, final_boxes, scores, class_ids, image_id)


        return self._format_coco(image_id, final_boxes, scores, class_ids, self.yolo_to_coco_id)