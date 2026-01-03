from ultralytics import YOLO
import numpy as np
import cv2
import csv

"""
Fichier : ball_recog.py (nom indicatif)

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquières
Date : 2025-11-27

Description :
Ce script applique un modèle YOLOv8 entraîné afin de détecter le ballon image par image
dans une vidéo provenant de la caméra 2.

Il produit :
1) Une vidéo annotée (trajectoire 2D + point courant),
2) Un fichier CSV contenant la piste 2D du ballon :
   [frame, time_s, x_px, y_px, confidence]

Ce CSV est ensuite utilisé pour la triangulation stéréo (reconstruction 3D).

Lien avec le pipeline du projet :
- Détection du ballon image par image (caméra 2)
- Extraction des coordonnées 2D (pixels) avec un score de confiance
- Préparation des données d’entrée pour la triangulation 3D (avec la caméra 1)
"""

if __name__ == "__main__":
    """
    Point d’entrée du script.

    Pipeline :
    1) Charger le modèle YOLOv8 (poids entraînés),
    2) Ouvrir la vidéo cam2,
    3) Pour chaque frame :
       - inférence YOLO,
       - filtrage classe ballon,
       - sauvegarde de toutes les détections (CSV),
       - sélection du meilleur candidat (confiance max) pour la trajectoire affichée,
       - annotation de l’image (polyligne + cercle),
       - écriture de la frame dans la vidéo de sortie,
    4) Export du CSV de tracking pour cam2.
    """

    # Chargement du modèle entraîné (poids du run Ultralytics)
    model = YOLO("runs/detect/train/weights/best.pt")  # or path to last.pt

    # ID de la classe "ballon" dans le dataset d'entraînement
    BALL_CLASS_ID = 0

    # Vidéo d’entrée (caméra 2)
    video_path = "path/to/test3D_cam2.mp4"
    cap = cv2.VideoCapture(video_path)

    # Vérification : arrêt immédiat si la vidéo ne peut pas être ouverte
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Paramètres vidéo (utile pour temps et écriture)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Préparation de la vidéo de sortie annotée
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("cam2_trajectory.mp4", fourcc, fps, (width, height))

    # data : stocke toutes les détections de ballon (une ligne par bbox)
    data = []

    # trajectory_points : stocke uniquement le meilleur centre par frame (visualisation trajectoire)
    trajectory_points = []

    # Index de frame (sert à calculer time_s et garder la synchronisation)
    frame_idx = 0

    while True:
        # Lecture de la frame suivante
        ret, frame = cap.read()
        if not ret:
            break

        # Inference YOLO sur la frame
        results = model.predict(frame, verbose=False)[0]
        boxes = results.boxes

        # Centre retenu pour cette frame (meilleur candidat)
        ball_center = None

        # Si des boîtes existent, chercher celles correspondant au ballon
        if boxes is not None and len(boxes) > 0:
            best_conf = 0.0
            best_center = None

            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                # Filtrage : ne garder que la classe ballon
                if int(cls) != BALL_CLASS_ID:
                    continue

                # Coordonnées bbox en pixels
                x1, y1, x2, y2 = box.tolist()

                # Centre du ballon (approximation par centre de la bbox)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Sélection du meilleur candidat (confiance max) pour affichage trajectoire
                if conf > best_conf:
                    best_conf = float(conf)
                    best_center = (cx, cy)

                # Conversion frame -> temps (secondes)
                t_sec = frame_idx / fps

                # Sauvegarde de la détection pour export CSV
                data.append([frame_idx, t_sec, cx, cy, float(conf)])

            # Ajout du meilleur centre dans la trajectoire (si trouvé)
            if best_center is not None:
                ball_center = best_center
                trajectory_points.append(ball_center)

        # Tracé de la trajectoire 2D (visualisation)
        if len(trajectory_points) >= 2:
            pts = np.array(trajectory_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

        # Dessin du ballon courant (meilleur centre)
        if ball_center is not None:
            cx, cy = ball_center
            cv2.circle(frame, (int(cx), int(cy)), radius=6, color=(0, 0, 255), thickness=-1)

        # Écriture de la frame annotée
        out.write(frame)

        frame_idx += 1

    # Libération des ressources OpenCV
    cap.release()
    out.release()

    # NOTE : message console conservé tel quel (même s'il ne correspond pas au nom exact)
    print("Done! Saved video as match_with_trajectory.mp4")

    # Export CSV : piste 2D cam2 (entrée de la triangulation)
    with open("ball_positions_csv_cam2", "w", newline="") as fwrite:
        writer = csv.writer(fwrite)
        writer.writerow(["frame", "time_s", "x_px", "y_px", "confidence"])
        writer.writerows(data)

    # NOTE : message console conservé tel quel (duplication de mots)
    print("Saved ball positions to ball positions to ball_positions.csv")


    

