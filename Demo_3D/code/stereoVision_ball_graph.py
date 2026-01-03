import sys
import time
import cv2
import numpy as np

# --- Stéréo / profondeur ---
import triangulation as tri
import calibration

# --- Hack WindowsPath (modèle YOLO entraîné sur Windows) ---
import pathlib
class _FakeWindowsPath(pathlib.PosixPath):
    pass
pathlib.WindowsPath = _FakeWindowsPath

# --- YOLO ---
import torch
from ultralytics import YOLO

# ================== PARAMÈTRES GLOBAUX ==================

# Charger ton modèle YOLO corrigé
model = YOLO('../yolo_models/best_mac2.pt')   # adapte le chemin si besoin
BALL_CLASS_ID = 0              # id de classe du ballon dans ton dataset

# Device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print("Using device:", device)
model.to(device)

# Caméras
cap_right = cv2.VideoCapture(0)
cap_left  = cv2.VideoCapture(1)

cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

retR, frame_right_init = cap_right.read()
retL, frame_left_init  = cap_left.read()
if not retR or not retL:
    raise RuntimeError("Impossible de lire une image d'une des caméras.")

print("Right frame shape:", frame_right_init.shape)
print("Left frame shape:",  frame_left_init.shape)

# Paramètres stéréo
B = 13.5   # [cm]
f = 8      # [mm]
alpha = 78 # [deg]

# Résolution YOLO réduite pour la vitesse
DETECT_W = 640
DETECT_H = 360

# Trajectoire (liste de (X, Y, Z) en cm)
trajectory = []


# ================== FONCTIONS YOLO ==================

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

    mask = (cls == class_id)
    if not np.any(mask):
        return None

    idxs = np.where(mask)[0]
    best_local = idxs[np.argmax(conf[mask])]
    x1, y1, x2, y2 = xyxy[best_local]

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return (cx, cy), (int(x1), int(y1), int(x2), int(y2))


# ================== FONCTIONS TRAJECTOIRE 3D (OpenCV) ==================

def project_point_3d_to_2d(X, Y, Z, origin, scale=0.01):
    """
    Projette un point 3D (en cm) dans un plan 2D pour affichage.
    Projection oblique simple pour donner un effet 3D.
    - X : gauche/droite
    - Y : haut/bas (déjà "monde", donc Y>0 vers le haut)
    - Z : profondeur
    origin : (ox, oy) = centre du repère dans l'image
    scale : facteur [OpenCV px] / [cm]
    """
    ox, oy = origin
    # Petite projection isométrique / oblique
    u = int(ox + scale * ( 0.7 * X - 0.4 * Z))
    v = int(oy - scale * ( 0.7 * Y + 0.4 * Z))
    return u, v


def draw_trajectory_3d_image(traj, size=(600, 600)):
    """
    Dessine un "graphique 3D" simplifié (projection 2D) dans une image OpenCV.
    traj : liste de (X, Y, Z) en cm (Y déjà orienté vers le haut)
    size : (h, w) de l'image résultat
    """
    h, w = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Origine du repère (un peu en bas de l'image)
    origin = (w // 2, h // 2 + 80)

    # Axes 3D (X rouge, Y vert, Z bleu) avec échelle fixe
    axis_len = 200.0  # [cm] longueur d'axe
    scale_axis = 0.01 # même ordre de grandeur que dans project_point_3d_to_2d

    X_axis_end = project_point_3d_to_2d(axis_len,   0.0,        0.0,        origin, scale_axis)
    Y_axis_end = project_point_3d_to_2d(0.0,        axis_len,   0.0,        origin, scale_axis)
    Z_axis_end = project_point_3d_to_2d(0.0,        0.0,        axis_len,   origin, scale_axis)

    # X axis
    cv2.arrowedLine(canvas, origin, X_axis_end, (0, 0, 255), 2, tipLength=0.08)
    cv2.putText(canvas, "X", (X_axis_end[0] + 5, X_axis_end[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Y axis
    cv2.arrowedLine(canvas, origin, Y_axis_end, (0, 255, 0), 2, tipLength=0.08)
    cv2.putText(canvas, "Y", (Y_axis_end[0] + 5, Y_axis_end[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Z axis
    cv2.arrowedLine(canvas, origin, Z_axis_end, (255, 0, 0), 2, tipLength=0.08)
    cv2.putText(canvas, "Z", (Z_axis_end[0] + 5, Z_axis_end[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Trajectoire
    if len(traj) > 1:
        prev_pt = None
        for (X, Y, Z) in traj:
            u, v = project_point_3d_to_2d(X, Y, Z, origin, scale=0.01)
            if prev_pt is not None:
                cv2.line(canvas, prev_pt, (u, v), (0, 255, 255), 2)
            prev_pt = (u, v)

        # Dernier point en cercle
        cv2.circle(canvas, prev_pt, 6, (0, 0, 255), -1)

    return canvas


# ================== BOUCLE PRINCIPALE ==================

while True:

    succes_right, frame_right = cap_right.read()
    succes_left,  frame_left  = cap_left.read()
    if not succes_right or not succes_left:
        print("Lecture caméra impossible, arrêt.")
        break

    start = time.time()

    # --- YOLO sur images réduites ---
    small_right = cv2.resize(frame_right, (DETECT_W, DETECT_H))
    small_left  = cv2.resize(frame_left,  (DETECT_W, DETECT_H))

    results = model([small_right, small_left], conf=0.4, verbose=False)
    yolo_res_right = results[0]
    yolo_res_left  = results[1]

    center_point_right = None
    center_point_left  = None

    # Caméra droite
    out_right = get_ball_center_yolo(yolo_res_right, BALL_CLASS_ID)
    if out_right is not None:
        (cx_s, cy_s), (x1_s, y1_s, x2_s, y2_s) = out_right
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

    # Caméra gauche
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

    # --- Calcul X, Y, Z ---
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

        # On stocke avec Y inversé pour avoir Y>0 vers le haut dans le graphe
        trajectory.append((X, -Y, Z))
        if len(trajectory) > 500:
            trajectory.pop(0)

        txt = f"X: {X:.1f} cm  Y: {Y:.1f} cm  Z: {Z:.1f} cm"
        cv2.putText(frame_right, txt, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame_left, txt, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    else:
        cv2.putText(frame_right, "BALL TRACKING LOST", (75, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_left, "BALL TRACKING LOST", (75, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- Fenêtre Stereo ---
    combined = np.hstack((frame_right, frame_left))

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime if totalTime > 0 else 0
    cv2.putText(combined, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Stereo", combined)

    # --- Fenêtre Trajectory 3D ---
    traj_img = draw_trajectory_3d_image(trajectory, size=(600, 600))
    cv2.imshow("3D Trajectory", traj_img)

    # Quitter si on ferme la fenêtre ou on appuie 'q'
    if cv2.getWindowProperty("Stereo", cv2.WND_PROP_VISIBLE) < 1:
        print("Fenêtre Stereo fermée.")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Touche 'q' pressée, arrêt.")
        break

# Nettoyage
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()




