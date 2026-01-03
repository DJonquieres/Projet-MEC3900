import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# >>> Hack WindowsPath AVANT d'importer ultralytics <<<
import pathlib
class _FakeWindowsPath(pathlib.PosixPath):
    pass
pathlib.WindowsPath = _FakeWindowsPath
# >>> Fin du hack <<<

import torch
from ultralytics import YOLO

# Charger le modèle corrigé
model = YOLO('../yolo_models/best_mac2.pt')
BALL_CLASS_ID = 0  # id de classe du ballon dans ton dataset

# Choix du device (GPU M1 si dispo)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print("Using device:", device)
model.to(device)

# Open both cameras
cap_right = cv2.VideoCapture(0)
cap_left  = cv2.VideoCapture(1)

cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

retR, frame_right = cap_right.read()
retL, frame_left  = cap_left.read()
print("Right frame shape:", frame_right.shape)
print("Left frame shape:",  frame_left.shape)

# Stereo vision setup parameters
frame_rate = 120
B = 13.5   # [cm]
f = 8      # [mm] (pas utilisé dans tri)
alpha = 78 # [deg]

# Résolution utilisée pour la détection YOLO (plus petite)
DETECT_W = 640
DETECT_H = 360


def get_ball_center_yolo(result, class_id=BALL_CLASS_ID):
    """
    result : résultat YOLO pour UNE image (model(...)[i])
    Retourne ((cx, cy), (x1, y1, x2, y2)) dans l'image DÉTECTÉE si trouvé, sinon None.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    # Filtrer uniquement la classe ballon
    mask = (cls == class_id)
    if not np.any(mask):
        return None

    # Prendre la détection la plus confiante pour le ballon
    idxs = np.where(mask)[0]
    best_local = idxs[np.argmax(conf[mask])]
    x1, y1, x2, y2 = xyxy[best_local]

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return (cx, cy), (int(x1), int(y1), int(x2), int(y2))


while cap_right.isOpened() and cap_left.isOpened():

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left   = cap_left.read()

    # If cannot catch any frame, break
    if not succes_right or not succes_left:
        break

    start = time.time()

    ################## DÉTECTION YOLO DU BALLON (RAPIDE) ######################

    # On crée des versions réduites pour la détection
    small_right = cv2.resize(frame_right, (DETECT_W, DETECT_H))
    small_left  = cv2.resize(frame_left,  (DETECT_W, DETECT_H))

    # Une seule prédiction YOLO sur les deux images
    results = model([small_right, small_left], conf=0.4, verbose=False, device=device)
    yolo_res_right = results[0]
    yolo_res_left  = results[1]

    center_point_right = None
    center_point_left  = None

    # --- Caméra droite ---
    out_right = get_ball_center_yolo(yolo_res_right, BALL_CLASS_ID)
    if out_right is not None:
        (cx_s, cy_s), (x1_s, y1_s, x2_s, y2_s) = out_right

        # remettre les coordonnées dans l'image originale (1280x720)
        Hr, Wr, _ = frame_right.shape
        scale_x = Wr / DETECT_W
        scale_y = Hr / DETECT_H

        cx = int(cx_s * scale_x)
        cy = int(cy_s * scale_y)
        x1 = int(x1_s * scale_x)
        y1 = int(y1_s * scale_y)
        x2 = int(x2_s * scale_x)
        y2 = int(y2_s * scale_y)

        center_point_right = (cx, cy)
        cv2.rectangle(frame_right, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame_right, center_point_right, 5, (0, 0, 255), -1)

    # --- Caméra gauche ---
    out_left = get_ball_center_yolo(yolo_res_left, BALL_CLASS_ID)
    if out_left is not None:
        (cx_s, cy_s), (x1_s, y1_s, x2_s, y2_s) = out_left

        Hl, Wl, _ = frame_left.shape
        scale_x = Wl / DETECT_W
        scale_y = Hl / DETECT_H

        cx = int(cx_s * scale_x)
        cy = int(cy_s * scale_y)
        x1 = int(x1_s * scale_x)
        y1 = int(y1_s * scale_y)
        x2 = int(x2_s * scale_x)
        y2 = int(y2_s * scale_y)

        center_point_left = (cx, cy)
        cv2.rectangle(frame_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame_left, center_point_left, 5, (0, 0, 255), -1)

    ################## CALCUL DU X, Y, Z APPROX ###############################

    if (center_point_right is not None) and (center_point_left is not None):

        depth = tri.find_depth(center_point_right, center_point_left,
                               frame_right, frame_left, B, f, alpha)

        h, w, _ = frame_right.shape
        f_pixel = (w * 0.5) / np.tan(alpha * 0.5 * np.pi / 180.0)
        cx_img = w / 2.0
        cy_img = h / 2.0

        u, v = center_point_right

        X = (u - cx_img) * depth / f_pixel
        Y = (v - cy_img) * depth / f_pixel
        Z = depth

        txt = f"X: {X:.1f} cm  Y: {Y:.1f} cm  Z: {Z:.1f} cm"
        cv2.putText(frame_right, txt, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame_left, txt, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        print(f"Depth (Z): {depth:.1f} cm  X: {X:.1f} cm  Y: {Y:.1f} cm")

    else:
        cv2.putText(frame_right, "BALL TRACKING LOST", (75, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_left, "BALL TRACKING LOST", (75, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    ################## FPS ####################################################

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime if totalTime > 0 else 0

    cv2.putText(frame_right, f'FPS: {int(fps)}', (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame_left, f'FPS: {int(fps)}', (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("frame right", frame_right)
    # cv2.imshow("frame left", frame_left)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()

