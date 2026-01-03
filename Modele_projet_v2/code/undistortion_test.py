import cv2
import numpy as np
from zhang_calibration import *

"""
Fichier : undistortion_test.py 

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-11-12

Description :
Ce script sert à évaluer qualitativement et quantitativement la calibration intrinsèque
(et surtout la correction de distorsion) d’une caméra.

Principe :
- On calibre la caméra (K, dist) à partir d’un dossier d’images de damier (méthode de Zhang).
- Sur une image test contenant un damier, on détecte les coins puis on les "undistort".
- On mesure à quel point une rangée de coins est "droite" :
  - avant correction (image originale),
  - après correction (points undistortés).
Une diminution de l’erreur indique une meilleure correction de distorsion.

Lien avec le pipeline du projet :
- Calibration intrinsèque (K, dist) nécessaire pour :
  - undistortion des frames,
  - conversion pixel -> coordonnées normalisées,
  - triangulation stéréo 3D plus fiable.
"""


def straightness_error(pts_row):
    """
    Calcule une métrique de "rectitude" d’une rangée de points.

    Méthode :
    - On ajuste une droite y = a*x + b (régression linéaire via polyfit)
    - On calcule l’erreur RMS des résidus (écarts verticals à la droite)

    Paramètres
    ----------
    pts_row : np.ndarray (N x 2)
        Points 2D (x, y) représentant une rangée du damier.

    Retour
    ------
    float
        Erreur RMS (en pixels si les points sont en pixels).
        Plus cette valeur est faible, plus les points sont alignés.
    """
    xs = pts_row[:, 0]
    ys = pts_row[:, 1]

    # Ajustement linéaire (ordre 1) : modèle de droite
    a, b = np.polyfit(xs, ys, 1)
    y_fit = a * xs + b

    # RMS des résidus (mesure simple de non-linéarité / courbure)
    return np.sqrt(np.mean((ys - y_fit) ** 2))


def test_error(path, path_test, CHECKERBOARD, ):
    """
    Teste l’amélioration de rectitude des coins d’un damier après undistortion.

    Étapes :
    1) Calibration intrinsèque (K, dist) à partir d’un dossier d’images (path).
    2) Lecture d’une image test (path_test).
    3) Détection des coins du damier sur l’image test.
    4) Raffinement subpixel (cornerSubPix) pour précision.
    5) Undistortion des coins (undistortPoints).
    6) Calcul de l’erreur de rectitude pour une rangée (avant / après).

    Paramètres
    ----------
    path : str
        Dossier d’images de calibration (intrinsèques) pour une caméra.
    path_test : str
        Chemin vers l’image test contenant un damier.
    CHECKERBOARD : tuple (cols, rows)
        Dimensions internes du damier (nombre de coins détectables).

    Retour
    ------
    ret : bool
        True si le damier est détecté dans l’image test, sinon False.
    error_orig : float
        Erreur de rectitude sur les points originaux (image distordue).
    error_und : float
        Erreur de rectitude sur les points undistortés.

    Notes
    -----
    Ce test est un indicateur pratique : si la correction est bonne,
    les rangées/colonnes du damier doivent se rapprocher d’une ligne droite.
    """
    # Calibration intrinsèque (méthode Zhang) : estimation de K et dist
    ret, K, dist, rvecs, tvecs = calibration(path, CHECKERBOARD)

    # Lecture image test
    img = cv2.imread(path_test)

    # Conversion en niveaux de gris (requis par findChessboardCorners)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection des coins du damier
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # Raffinement subpixel : améliore précision des coins avant l'évaluation
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # undistortPoints : correction de distorsion
        # P=K permet de récupérer des points dans le repère pixel (plutôt que normalisé)
        undistorted_pts = cv2.undistortPoints(corners, K, dist, P=K)
        undistorted_pts = undistorted_pts.reshape(-1, 2)

        # Sélection d’une rangée de coins pour l’évaluation
        # Ici : la première rangée (row0) correspondant aux CHECKERBOARD[0] premiers points
        row0_orig = corners[0:CHECKERBOARD[0], 0, :]
        row0_und = undistorted_pts[0:CHECKERBOARD[0], :]

        # Calcul des erreurs de rectitude
        error_orig = straightness_error(row0_orig)
        error_und = straightness_error(row0_und)

    # Retourne la détection et les métriques.
    # Si ret==False, error_orig/error_und ne seront pas définies (comportement actuel conservé).
    return ret, error_orig, error_und


# -------------------------------------------------------------------
# Paramètres du test (à adapter selon l’organisation du projet)
# -------------------------------------------------------------------

CHECKERBOARD = (9, 6)

# Dossiers de calibration intrinsèque pour chaque caméra
path1 = "calib_images/cam1"
path2 = "calib_images/cam2"

# Images test indépendantes (une par caméra)
path_test1 = "path/to/testimg.jpg"
path_test2 = "path/to/testimg2.jpg"

# Évaluation caméra 1 et caméra 2
ret1, error_orig1, error_und1 = test_error(path1, path_test1, CHECKERBOARD)
ret2, error_orig2, error_und2 = test_error(path2, path_test2, CHECKERBOARD)

# -------------------------------------------------------------------
# Affichage des résultats si les deux tests ont réussi
# -------------------------------------------------------------------

if ret1 and ret2:
    print(f"Straightness error 1 (original): {error_orig1}px")
    print(f"Straightness error 1 (undistorted): {error_und1}px")
    print(f"Straightness error 2 (original): {error_orig2}px")
    print(f"Straightness error 2 (undistorted): {error_und2}px")
 