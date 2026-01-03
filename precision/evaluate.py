import os
import json
import csv
import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

# --- Paramètres ---
coco_annotations = "../data/annotations/instances_val2017.json"
image_dir = "../data/val2017/"

def evaluate(model, output_json, limit=None):
    coco = COCO(coco_annotations)

    # Initialisation modèle
    if not os.path.exists(output_json):
        """ sess = ort.InferenceSession(
            model.path, providers=ort.get_available_providers()) """

        img_ids = coco.getImgIds()
        if limit:
            img_ids = img_ids[:limit]

        results = []
        for img_id in tqdm(img_ids, desc="Inférence"):
            #img_id = 204329
            info = coco.loadImgs(img_id)[0]
            image_path = os.path.join(image_dir, info['file_name'])
            if not os.path.exists(image_path):
                continue

            image = cv2.imread(image_path)
            """ results += model.run(image, sess, img_id) """
            results += model.run(image, img_id)


        # Sauvegarde des résultats au format COCO
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f)
        print(f"Résultats sauvegardés -> {output_json}")

    # Évaluation avec pycocotools
    cocoDt = coco.loadRes(output_json)
    cocoEval = COCOeval(coco, cocoDt, iouType="bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = cocoEval.params
        typeStr = 'AP' if ap==1 else 'AR'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = cocoEval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = cocoEval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])

        os.makedirs("./results", exist_ok=True)

        if not os.path.exists('./results/scores.csv'):
            with open('./results/scores.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['model', 'type', 'dataset', 'metric', 'IoU', 'areaRng', 'maxDets', 'score'])

        df = pd.read_csv('./results/scores.csv', index_col=['model', 'type', 'dataset', 'metric', 'IoU', 'areaRng', 'maxDets'])
        df.loc[(model.name, model.type, "validation", typeStr, iouStr, areaRng, maxDets), :] = float(mean_s)
        df.to_csv('./results/scores.csv')

        return mean_s
    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=cocoEval.params.maxDets[2])
        stats[1] = _summarize(0)
        return stats
    _summarizeDets()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=False, help="Chemin vers le modèle TFLite (.tflite)")
    parser.add_argument("--model-name", required=True, help="Nom du modèle")
    parser.add_argument("--type", required=True, help="Format du modèle: onnx, pytorch")
    parser.add_argument("--output", default="results.json", help="Fichier de sortie COCO")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre d'images pour les tests")
    args = parser.parse_args()

    if args.model_name == "yolov5" and args.type == "onnx":
        from models.yolov5 import YoloV5ONNXModel
        model = YoloV5ONNXModel(
            path=args.model_path,
            name="yolov5"
        )
    elif args.model_name == "yolov5" and args.type == "pytorch":
        from models.yolov5 import YoloV5UltralyticsModel
        import torch
        model = YoloV5UltralyticsModel(
            name="yolov5",
            path=args.model_path,
            input_shape=(640, 640)
        )
    elif args.model_name == "yolov11" and args.type == "onnx":
        from models.yolov11 import YoloV11ONNXModel
        model = YoloV11ONNXModel(
            path=args.model_path,
            name="yolov11"
        )
    elif args.model_name == "yolov11" and args.type == "pytorch":
        from models.yolov11 import YoloV11UltralyticsModel
        import torch
        model = YoloV11UltralyticsModel(
            name="yolov11",
            path=args.model_path,
            input_shape=(416, 416)
        )
    elif args.model_name == "fasterRCNN" and args.type == "pytorch":
        from models.fasterRCNN import FasterRCNNPytorchModel
        import torch
        model = FasterRCNNPytorchModel(
            name="fasterRCNN",
            input_shape=(320, 320),
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    elif args.model_name == "ssd" and args.type == "onnx":
        from models.ssd import SSDMobileNetONNXModel
        model = SSDMobileNetONNXModel(
            path=args.model_path,
            name="ssd"
        )
    elif args.model_name == "ssd" and args.type == "pytorch":
        from models.ssd import SSDMobileNetPytorchModel
        import torch
        model = SSDMobileNetPytorchModel(
            name="ssd",
            input_shape=(320, 320),
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    else:
        raise NotImplementedError(f"{args.model_name} model is not implemented.")
    
    evaluate(model, args.output, limit=args.limit)