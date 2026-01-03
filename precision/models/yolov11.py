import cv2
import numpy as np

from models.model import ONNXModel, UltralyticsModel

class YoloV11ONNXModel(ONNXModel):
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

    def postprocess(self, outputs, image, image_id, conf_thres=0.0001, iou_thres=0.45):
        """
        Décodage spécifique YOLOv11 : Gestion du tenseur transposé [1, 4+C, 8400].
        Source: [15, 17]
        """
        # Transposition critique :  -> 
        # On suppose 80 classes -> 4 coords + 80 classes = 84 channels
        predictions = np.squeeze(outputs).T
        
        # Séparation des coordonnées et des scores
        # Les 4 premières colonnes sont cx, cy, w, h
        box_data = predictions[:, :4]
        # Les colonnes suivantes sont les probabilités de classe
        class_data = predictions[:, 4:]

        # Calcul vectorisé des scores max et des IDs de classe
        # Pas de score 'objectness' séparé dans v8/v11 export par défaut
        max_scores = np.max(class_data, axis=1)
        argmax_ids = np.argmax(class_data, axis=1)

        # Filtrage par seuil de confiance
        mask = max_scores > conf_thres
        filtered_boxes = box_data[mask]
        filtered_scores = max_scores[mask]
        filtered_ids = argmax_ids[mask]
        
        if len(filtered_boxes) == 0:
            return []

        # Conversion et redimensionnement
        scale, pad_left, pad_top = self.pad_params
        
        # Copie pour éviter de modifier en place si nécessaire
        final_boxes_coords = []
        
        for i in range(len(filtered_boxes)):
            cx, cy, w, h = filtered_boxes[i]
            
            # Inversion du letterbox
            # On retire le padding puis on divise par le scale
            cx = (cx - pad_left) / scale
            cy = (cy - pad_top) / scale
            w = w / scale
            h = h / scale
            
            # Conversion Center -> Top-Left pour COCO
            x = cx - (w / 2)
            y = cy - (h / 2)
            
            final_boxes_coords.append([x, y, w, h])

        # NMS
        final_boxes_coords = np.array(final_boxes_coords)
        indices = self._nms(final_boxes_coords, filtered_scores, iou_thres)
        
        return self._format_coco(
            image_id,
            final_boxes_coords[indices], 
            filtered_scores[indices], 
            filtered_ids[indices],
            self.yolo_to_coco_id
        )
    
class YoloV11UltralyticsModel(UltralyticsModel):

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