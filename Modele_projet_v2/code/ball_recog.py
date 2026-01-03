from ultralytics import YOLO
import numpy as np
import cv2
import csv

"""
Fichier : ball_recog.py

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-11-25

Description :
Ce script applique un modèle YOLOv8 entraîné sur des images de volleyball afin de :
1) Détecter le ballon image par image dans une vidéo,
2) Extraire ses coordonnées 2D (en pixels) et la confiance associée,
3) Enregistrer une vidéo annotée (trajectoire + position courante),
4) Exporter un CSV contenant la piste 2D du ballon pour utilisation ultérieure
   (ex : triangulation 3D / modélisation de trajectoire / prédiction du point d’impact).

Lien avec les fonctions du projet :
- FP : Détection du ballon image par image
- FP : Extraction des coordonnées (pixels) et conversion ultérieure (vers 3D)
- Besoins implicites : robustesse (confiance), sortie exploitable (CSV), code compréhensible

Dépendances principales :
- Ultralytics YOLO (inférence)
- OpenCV (lecture/écriture vidéo + dessin)
- NumPy (gestion des points de trajectoire)
"""

if __name__ == "__main__":
    """
    Point d’entrée du script.

    Pipeline exécuté :
    1) Charger le modèle YOLOv8 (poids entraînés).
    2) Ouvrir la vidéo à analyser.
    3) Pour chaque frame :
       - Exécuter l’inférence YOLO,
       - Filtrer la classe "ballon",
       - Enregistrer toutes les détections (frame/time/x/y/conf),
       - Choisir la meilleure détection (confiance max) pour tracer la trajectoire,
       - Annoter l’image (polyligne + point),
       - Écrire la frame annotée dans la vidéo de sortie.
    4) Exporter les positions détectées dans un CSV.

    Remarques importantes :
    - On définit BALL_CLASS_ID selon le dataset utilisé à l’entraînement.
    - Le CSV est un format d’échange : il sera réutilisé par la calibration/triangulation stéréo.
    """

    # Chargement du modèle YOLO (poids "best.pt" après entraînement)
    # Pourquoi "best.pt" ? En général, c’est le checkpoint avec les meilleures métriques sur validation.
    model = YOLO("runs/detect/train2/weights/best.pt")  

    # ID de classe correspondant au ballon dans le dataset d'entraînement
    # (à maintenir cohérent avec la configuration du dataset)
    BALL_CLASS_ID = 0

    # Chemin vers la vidéo à analyser
    video_path = "path/to/Video Project 3.mp4"
    cap = cv2.VideoCapture(video_path)

    # Vérification : si la vidéo ne s’ouvre pas, on arrête immédiatement (erreur claire)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Récupération des paramètres vidéo (nécessaires pour synchronisation temps + écriture sortie)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Paramétrage de la vidéo de sortie (frames annotées)
    # Codec mp4v : compromis simple/portable avec OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("cam1_trajectory.mp4", fourcc, fps, (width, height))

    # data : stocke TOUTES les détections de ballon (une ligne par bbox classée ballon)
    # Format : [frame_idx, time_s, x_px, y_px, confidence]
    data = []

    # trajectory_points : stocke uniquement les meilleurs centres (meilleure confiance par frame)
    # Utilisé pour tracer une trajectoire visuelle plus propre sur la vidéo.
    trajectory_points = []

    # Index de frame (sert à calculer le temps et à conserver l’alignement avec la vidéo)
    frame_idx = 0

    while True:
        # Lecture de la frame suivante
        ret, frame = cap.read()
        if not ret:
            # Fin de la vidéo ou erreur de lecture -> sortie de boucle
            break

        # Inference YOLO sur la frame
        # verbose=False pour éviter de saturer la console pendant le traitement
        results = model.predict(frame, verbose=False)[0]
        boxes = results.boxes

        # ball_center : centre retenu pour cette frame (meilleur candidat)
        ball_center = None

        # Vérification de la présence de détections
        if boxes is not None and len(boxes) > 0:
            best_conf = 0.0
            best_center = None

            # Parcours de toutes les boîtes prédites
            # On garde uniquement celles de la classe "ballon"
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                if int(cls) != BALL_CLASS_ID:
                    continue

                # Coordonnées bbox (x1,y1,x2,y2) en pixels
                x1, y1, x2, y2 = box.tolist()

                # Calcul du centre (cx, cy) : représentation simple et adaptée au suivi
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Sélection du "meilleur" ballon sur cette frame (confiance maximale)
                # Pourquoi ? Réduit l’impact de faux positifs si plusieurs objets ressemblent au ballon.
                if conf > best_conf:
                    best_conf = float(conf)
                    best_center = (cx, cy)

                # Conversion frame -> temps (secondes)
                # Hypothèse : fps constant sur la vidéo
                t_sec = frame_idx / fps

                # Enregistrement de la détection dans la sortie tabulaire (CSV)
                # Note : ici on stocke chaque détection "ballon" trouvée sur la frame
                data.append([frame_idx, t_sec, cx, cy, float(conf)])

            # Si une détection a été retenue, on la conserve pour la trajectoire dessinée
            if best_center is not None:
                ball_center = best_center
                trajectory_points.append(ball_center)

        # Tracé de la trajectoire (si au moins 2 points)
        # NOTE : la trajectoire est en pixels (2D). Elle servira ensuite de piste pour la triangulation 3D.
        if len(trajectory_points) >= 2:
            pts = np.array(trajectory_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

        # Marqueur du ballon sur la frame actuelle (meilleur candidat)
        if ball_center is not None:
            cx, cy = ball_center
            cv2.circle(frame, (int(cx), int(cy)), radius=6, color=(0, 0, 255), thickness=-1)

        # Écriture de la frame annotée dans la vidéo de sortie
        out.write(frame)

        # Incrément de la frame
        frame_idx += 1

    # Libération des ressources (bonne pratique OpenCV : évite fichiers corrompus/locks)
    cap.release()
    out.release()

    # NOTE : message légèrement incohérent avec le nom réel du fichier généré.
    # On n’y touche pas car demandé : "sans le changer".
    print("Done! Saved video as match_with_trajectory.mp4")

    # Export CSV : format standardisé utilisé ensuite pour triangulation (frame, time_s, x_px, y_px, confidence)
    # IMPORTANT : le nom de fichier doit être cohérent avec le script de triangulation/lecture CSV.
    with open("ball_positions_csv_cam1", "w", newline="") as fwrite:
        writer = csv.writer(fwrite)
        writer.writerow(["frame", "time_s", "x_px", "y_px", "confidence"])
        writer.writerows(data)

    # Message console de fin (note : duplication "to ball positions to")
    # On n’y touche pas car demandé : "sans le changer".
    print("Saved ball positions to ball positions to ball_positions.csv")


    

