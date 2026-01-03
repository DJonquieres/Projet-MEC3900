from ultralytics import YOLO
from pathlib import Path

"""
Fichier : ball_model_training_v2.py

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-10-23

Description :
Ce script lance l’entraînement d’un modèle de détection YOLOv8 à partir d’un dataset
personnalisé (ballon de volleyball).

Principe :
- Un modèle YOLOv8 pré-entraîné est utilisé comme point de départ (transfer learning),
- Le dataset est décrit via un fichier data.yaml (chemins + classes),
- Les hyperparamètres principaux (taille d’image, epochs, batch, early stopping)
  sont définis pour obtenir un compromis précision / temps d’entraînement.

Lien avec le pipeline du projet :
- Étape d’apprentissage du détecteur de ballon
- Condition essentielle pour l’extraction fiable des coordonnées 2D (x_px, y_px)
- Les sorties de ce modèle alimentent ensuite la triangulation 3D et la modélisation
  de trajectoire
"""

if __name__ == "__main__":
    """
    Point d’entrée du script.

    Étapes exécutées :
    1) Chargement d’un modèle YOLOv8 pré-entraîné (poids initiaux),
    2) Chargement de la configuration du dataset (data.yaml),
    3) Lancement de l’entraînement avec les hyperparamètres définis,
    4) (Optionnel) Test du modèle entraîné sur une vidéo.
    """

    # Chargement du modèle YOLOv8
    model = YOLO('yolov8n')

    # Chemin vers le fichier data.yaml décrivant le dataset (train / validation / classes)
    datapath = Path('../data.yaml')

    # Lancement de l’entraînement
    model.train(
        data=str(datapath),  # configuration du dataset
        imgsz=640,           # taille des images d’entrée (standard YOLO)
        epochs=120,          # nombre maximum d’epochs
        batch=16,            # taille du batch (dépend de la mémoire GPU)
        patience=20          # early stopping : arrêt si pas d’amélioration
    )

    # -------------------------------------------------------------------
    # Section optionnelle : test du modèle entraîné sur une vidéo
    # -------------------------------------------------------------------
    # Ces lignes permettent de valider qualitativement les performances
    # du détecteur (visualisation des bounding boxes sur une vidéo).

    # model = YOLO("../runs/detect/train2/weights/best.pt")
    # results = model.predict("../path/to/test.mp4", show=True, save=True)


