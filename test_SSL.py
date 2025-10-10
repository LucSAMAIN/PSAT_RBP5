import cv2
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import datetime
import requests

# --- 1. Chargement du Modèle et des Classes ---

# Charger un modèle ResNet50 pré-entraîné. C'est un exemple de classifieur puissant.
# Remplacez-le par le chargement de votre modèle SSL spécifique si vous en avez un.
try:
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.eval() # Mettre le modèle en mode évaluation
except Exception:
    print("Could not load model weights automatically. Please ensure you have torchvision installed and an internet connection.")
    exit()


# Obtenir les noms des classes d'ImageNet pour afficher des résultats lisibles
try:
    response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    labels = response.text.split('\n')
except Exception:
    print("Could not download class labels. Predictions will be shown as numbers.")
    labels = [str(i) for i in range(1000)]


# Définir les transformations d'image nécessaires pour le modèle
# Le modèle s'attend à des images de 224x224 avec une normalisation spécifique.
preprocess = weights.transforms()


# --- 2. Configuration de la Vidéo ---

video_path = "/dev/video0"
videoCap = cv2.VideoCapture(video_path)

# Propriétés de la vidéo pour l'enregistrement
frame_width = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = videoCap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 20

# Codec et objet VideoWriter pour sauvegarder la vidéo
output_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_ssl.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


# --- 3. Boucle de Traitement Vidéo ---

while True:
    ret, frame = videoCap.read()
    if not ret:
        break

    # --- Inférence du modèle SSL (Classification) ---
    # 1. Convertir l'image OpenCV (BGR) en image PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 2. Appliquer les transformations et créer un "batch" de 1 image
    batch = preprocess(img_pil).unsqueeze(0)

    # 3. Faire la prédiction
    with torch.no_grad():
        prediction = model(batch).squeeze(0).softmax(0)

    # 4. Obtenir la classe avec le score le plus élevé
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    class_name = labels[class_id]

    # Afficher le résultat seulement si la confiance est suffisante
    if score > 0.2:
        text = f'{class_name}: {score:.2f}'
        # Mettre le texte en haut à gauche de l'image
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Sauvegarder l'image dans la vidéo de sortie
    out.write(frame)

    # Afficher l'image
    cv2.imshow("SSL Model Classification", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- 4. Libération des ressources ---
videoCap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vidéo sauvegardée sous : {output_path}")
