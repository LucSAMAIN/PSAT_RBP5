import cv2
import numpy as  np
import torch

from models.model import ONNXModel, PyTorchModel
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
import torchvision.models.detection as models

class SSDMobileNetONNXModel(ONNXModel):
    ssd_to_coco = {
        1: 0,    # Person
        2: 1,    # Bicycle
        3: 2,    # Car
        4: 3,    # Motorcycle
        5: 4,    # Airplane
        6: 5,    # Bus
        7: 6,    # Train
        8: 7,    # Truck
        9: 8,    # Boat
        10: 9,   # Traffic light
        11: 10,  # Fire hydrant
        13: 11,  # Stop sign
        14: 12,  # Parking meter
        15: 13,  # Bench
        16: 14,  # Bird
        17: 15,  # Cat
        18: 16,  # Dog
        19: 17,  # Horse
        20: 18,  # Sheep
        21: 19,  # Cow
        22: 20,  # Elephant
        23: 21,  # Bear
        24: 22,  # Zebra
        25: 23,  # Giraffe
        27: 24,  # Backpack
        28: 25,  # Umbrella
        31: 26,  # Handbag
        32: 27,  # Tie
        33: 28,  # Suitcase
        34: 29,  # Frisbee
        35: 30,  # Skis
        36: 31,  # Snowboard
        37: 32,  # Sports ball
        38: 33,  # Kite
        39: 34,  # Baseball bat
        40: 35,  # Baseball glove
        41: 36,  # Skateboard
        42: 37,  # Surfboard
        43: 38,  # Tennis racket
        44: 39,  # Bottle
        46: 40,  # Wine glass
        47: 41,  # Cup
        48: 42,  # Fork
        49: 43,  # Knife
        50: 44,  # Spoon
        51: 45,  # Bowl
        52: 46,  # Banana
        53: 47,  # Apple
        54: 48,  # Sandwich
        55: 49,  # Orange
        56: 50,  # Broccoli
        57: 51,  # Carrot
        58: 52,  # Hot dog
        59: 53,  # Pizza
        60: 54,  # Donut
        61: 55,  # Cake
        62: 56,  # Chair
        63: 57,  # Couch
        64: 58,  # Potted plant
        65: 59,  # Bed
        67: 60,  # Dining table
        70: 61,  # Toilet
        72: 62,  # Tv
        73: 63,  # Laptop
        74: 64,  # Mouse
        75: 65,  # Remote
        76: 66,  # Keyboard
        77: 67,  # Cell phone
        78: 68,  # Microwave
        79: 69,  # Oven
        80: 70,  # Toaster
        81: 71,  # Sink
        82: 72,  # Refrigerator
        84: 73,  # Book
        85: 74,  # Clock
        86: 75,  # Vase
        87: 76,  # Scissors
        88: 77,  # Teddy bear
        89: 78,  # Hair drier
    90: 79   # Toothbrush
}
    def preprocess(self, image):
        """
        Prétraitement SSD MobileNet (Style TensorFlow).
        Intervalle [-1, 1] et Resize simple.
        Source: 
        """
        # print(image.shape[:2])
        self.original_h, self.original_w = image.shape[:2]
        target_h, target_w = self.input_shapes[0] # ex: 300, 300
        
        resized = cv2.resize(image, (target_w, target_h))
        # Conversion BGR -> RGB
        # image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalisation [-1, 1] : (pixel - 127.5) / 127.5
        input_tensor = resized.astype(np.float32) / 255.0
        
        # TensorFlow utilise NHWC en interne, mais la conversion ONNX (tf2onnx) 
        # convertit souvent en NCHW pour la compatibilité. 
        # Nous assumons ici NCHW standard ONNX. Si le modèle est NHWC, supprimer le transpose.
        if self.session.get_inputs()[0].shape[1] == 3:
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        """ cv2.imshow("Preprocessed Input", np.transpose(input_tensor[0], (1, 2, 0)))
        cv2.waitKey(0) """

        return input_tensor
    
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

    def postprocess(self, outputs, image, image_id, conf_thres=0.0001):
        """
        Décodage SSD MobileNet (TF).
        Attention à l'ordre [y_min, x_min, y_max, x_max].
        Source: 
        """
        # 1. Unpack and Squeeze Batch Dimension
        # TF SSD Outputs: [Boxes (1, N, 4)], [Classes (1, N)], [Scores (1, N)]
        # print(outputs)
        raw_boxes = np.squeeze(outputs[0])  # Shape: (N, 4)
        raw_classes = np.squeeze(outputs[2]) # Shape: (N,)
        raw_scores = np.squeeze(outputs[1])  # Shape: (N,)

        # 2. Filter by Confidence
        mask = raw_scores > conf_thres
        boxes = raw_boxes[mask]
        scores = raw_scores[mask]
        class_ids = raw_classes[mask].astype(np.int32)

        if len(boxes) == 0:
            return []

        # 3. Coordinate Conversion
        # TF SSD native format is [y_min, x_min, y_max, x_max] (Normalized 0-1)
        # We need to extract them SAFELY into new variables
        y_min = boxes[:, 1]
        x_min = boxes[:, 0]
        y_max = boxes[:, 3]
        x_max = boxes[:, 2]

        # 4. Denormalize to Original Image Dimensions
        # Since they are 0-1, we multiply by original dimensions directly.
        # (We do NOT need scale_x/scale_y unless the model outputs absolute 300x300 coords)

        target_h, target_w = self.input_shapes[0]
        scale_y = self.original_h / target_h
        scale_x = self.original_w / target_w
        
        x1 = x_min * scale_x
        y1 = y_min * scale_y
        x2 = x_max * scale_x
        y2 = y_max * scale_y

        # 5. Convert to COCO [x, y, w, h] (Top-Left)
        final_boxes = np.zeros_like(boxes)
        final_boxes[:, 0] = x1          # x
        final_boxes[:, 1] = y1          # y
        final_boxes[:, 2] = x2 - x1     # w
        final_boxes[:, 3] = y2 - y1     # h

        # 6. Clip to image boundaries (Optional but safe)
        final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, self.original_w)
        final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, self.original_h)

        # print(final_boxes)

        # 7. Draw and Return
        # self.draw_boxes(image, final_boxes, scores, class_ids, image_id)

        return self._format_coco(image_id, final_boxes, scores, class_ids)


class SSDMobileNetPytorchModel(PyTorchModel):
    ssd_to_coco = {
        1: 0,    # Person
        2: 1,    # Bicycle
        3: 2,    # Car
        4: 3,    # Motorcycle
        5: 4,    # Airplane
        6: 5,    # Bus
        7: 6,    # Train
        8: 7,    # Truck
        9: 8,    # Boat
        10: 9,   # Traffic light
        11: 10,  # Fire hydrant
        13: 11,  # Stop sign
        14: 12,  # Parking meter
        15: 13,  # Bench
        16: 14,  # Bird
        17: 15,  # Cat
        18: 16,  # Dog
        19: 17,  # Horse
        20: 18,  # Sheep
        21: 19,  # Cow
        22: 20,  # Elephant
        23: 21,  # Bear
        24: 22,  # Zebra
        25: 23,  # Giraffe
        27: 24,  # Backpack
        28: 25,  # Umbrella
        31: 26,  # Handbag
        32: 27,  # Tie
        33: 28,  # Suitcase
        34: 29,  # Frisbee
        35: 30,  # Skis
        36: 31,  # Snowboard
        37: 32,  # Sports ball
        38: 33,  # Kite
        39: 34,  # Baseball bat
        40: 35,  # Baseball glove
        41: 36,  # Skateboard
        42: 37,  # Surfboard
        43: 38,  # Tennis racket
        44: 39,  # Bottle
        46: 40,  # Wine glass
        47: 41,  # Cup
        48: 42,  # Fork
        49: 43,  # Knife
        50: 44,  # Spoon
        51: 45,  # Bowl
        52: 46,  # Banana
        53: 47,  # Apple
        54: 48,  # Sandwich
        55: 49,  # Orange
        56: 50,  # Broccoli
        57: 51,  # Carrot
        58: 52,  # Hot dog
        59: 53,  # Pizza
        60: 54,  # Donut
        61: 55,  # Cake
        62: 56,  # Chair
        63: 57,  # Couch
        64: 58,  # Potted plant
        65: 59,  # Bed
        67: 60,  # Dining table
        70: 61,  # Toilet
        72: 62,  # Tv
        73: 63,  # Laptop
        74: 64,  # Mouse
        75: 65,  # Remote
        76: 66,  # Keyboard
        77: 67,  # Cell phone
        78: 68,  # Microwave
        79: 69,  # Oven
        80: 70,  # Toaster
        81: 71,  # Sink
        82: 72,  # Refrigerator
        84: 73,  # Book
        85: 74,  # Clock
        86: 75,  # Vase
        87: 76,  # Scissors
        88: 77,  # Teddy bear
        89: 78,  # Hair drier
    90: 79   # Toothbrush
}
    
    def __init__(self, name: str, input_shape: tuple = (640, 640), device: str = None):
        super().__init__(name, input_shape, device)

        print(f"Loading {self.name} on {self.device}...")

        self.model = models.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image):
        """
        Prétraitement SSD MobileNet (Style TensorFlow).
        Intervalle [-1, 1] et Resize simple.
        Source: 
        """
        # print(image.shape[:2])
        self.original_h, self.original_w = image.shape[:2]
        target_h, target_w = self.input_shapes[0] # ex: 300, 300

        img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = torch.from_numpy(img).to(self.device)

        img = img.permute(2, 0, 1)
        img /= 255.0
        img = img.unsqueeze(0)

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

    def postprocess(self, outputs, image, image_id, conf_thres=0.0001):
        """
        Décodage SSD MobileNet (TF).
        Attention à l'ordre [y_min, x_min, y_max, x_max].
        Source: 
        """
        # 1. Unpack and Squeeze Batch Dimension
        # TF SSD Outputs: [Boxes (1, N, 4)], [Classes (1, N)], [Scores (1, N)]
        # print(outputs)
        results = outputs[0]
        if not results or not 'boxes' in results or len(results['boxes']) == 0:
            return self._format_coco(image_id, [], [], [])
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        class_ids = results['labels'].cpu().numpy().astype(np.int32)

        # 2. Filter by Confidence
        mask = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask].astype(np.int32)

        if len(boxes) == 0:
            return self._format_coco(image_id, [], [], [])

        # 4. Denormalize to Original Image Dimensions
        # Since they are 0-1, we multiply by original dimensions directly.
        # (We do NOT need scale_x/scale_y unless the model outputs absolute 300x300 coords)

        target_h, target_w = self.input_shapes[0]
        scale_y = self.original_h / target_h
        scale_x = self.original_w / target_w

        x1 = boxes[:, 0] * scale_x
        y1 = boxes[:, 1] * scale_y
        x2 = boxes[:, 2] * scale_x
        y2 = boxes[:, 3] * scale_y

        # 5. Convert to COCO [x, y, w, h] (Top-Left)
        final_boxes = np.zeros_like(boxes)
        final_boxes[:, 0] = x1          # x
        final_boxes[:, 1] = y1          # y
        final_boxes[:, 2] = x2 - x1     # w
        final_boxes[:, 3] = y2 - y1     # h

        # 6. Clip to image boundaries (Optional but safe)
        final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, self.original_w)
        final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, self.original_h)

        # print(final_boxes)

        # 7. Draw and Return
        # self.draw_boxes(image, final_boxes, scores, class_ids, image_id)

        return self._format_coco(image_id, final_boxes, scores, class_ids)