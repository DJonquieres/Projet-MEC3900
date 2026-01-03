import cv2
import numpy as np
from zhang_calibration import *

"""
Fichier : undistortion_test.py (nom indicatif)

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-11-27

Description :
Ce script sert à évaluer la qualité de la calibration intrinsèque d’une caméra,
en mesurant l’amélioration de "rectitude" (straightness) d’une rangée de coins du damier
avant et après correction de distorsion.

Principe :
- On calibre la caméra (K, dist) à partir d’un dossier d’images de damier (méthode de Zhang).
- Sur une image test de damier, on détecte les coins.
- On applique l’undistortion aux coins détectés.
- On compare l’erreur de rectitude (RMS par rapport à une droite ajustée) :
  - sur les coins originaux (distordus),
  - sur les coins undistortés (corrigés).

Lien avec le pipeline du projet :
- Validation des paramètres intrinsèques (K, dist)
- Étape critique avant triangulation 3D : une mauvaise undistortion dégrade fortement la reconstruction.
"""


def straightness_error(pts_row):
    """
    Calcule une métrique simple de rectitude pour une rangée de points 2D.

    Méthode :
    - Ajuste une droite y = a*x + b (régression linéaire)
    - Calcule l’erreur RMS des écarts (ys - y_fit)

    Paramètres
    ----------
    pts_row : np.ndarray (N x 2)
        Points (x, y) correspondant à une rangée de coins de damier.

    Retour
    ------
    float
        Erreur RMS (en pixels si pts_row est en pixels).
        Plus l’erreur est faible, plus la rangée est proche d’une droite.
    """
    xs = pts_row[:, 0]
    ys = pts_row[:, 1]

    # Ajustement linéaire (droite) sur la rangée
    a, b = np.polyfit(xs, ys, 1)
    y_fit = a * xs + b

    # RMS des résidus : mesure la "courbure" résiduelle de la rangée
    return np.sqrt(np.mean((ys - y_fit) ** 2))


def test_error(path, path_test, CHECKERBOARD, ):
    """
    Évalue l’effet de l’undistortion sur une image test de damier.

    Paramètres
    ----------
    path : str
        Dossier d’images de calibration (intrinsèques) pour estimer K et dist.
    path_test : str
        Chemin vers une image test contenant le damier.
    CHECKERBOARD : tuple (cols, rows)
        Dimensions internes du damier.

    Retour
    ------
    ret : bool
        True si le damier est détecté dans l’image test, sinon False.
    error_orig : float
        Erreur de rectitude sur la rangée sélectionnée avant undistortion.
    error_und : float
        Erreur de rectitude sur la rangée sélectionnée après undistortion.

    Notes
    -----
    Si la calibration est correcte, on s’attend à :
        error_und < error_orig
    """
    # Calibration intrinsèque : estimation de K et dist
    ret, K, dist, rvecs, tvecs = calibration(path, CHECKERBOARD)

    # Lecture image test
    img = cv2.imread(path_test)

    # Conversion en niveaux de gris (requis pour findChessboardCorners)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection des coins du damier dans l’image test
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # Raffinement subpixel : améliore la précision des coins (comparaison plus fiable)
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # Correction de distorsion sur les points :
        # P=K renvoie des points dans le repère pixel (au lieu de coordonnées normalisées)
        undistorted_pts = cv2.undistortPoints(corners, K, dist, P=K)
        undistorted_pts = undistorted_pts.reshape(-1, 2)

        # Sélection d’une rangée de coins pour quantifier la rectitude
        # Ici : la première rangée (les CHECKERBOARD[0] premiers coins)
        row0_orig = corners[0:CHECKERBOARD[0], 0, :]
        row0_und = undistorted_pts[0:CHECKERBOARD[0], :]

        # Calcul des erreurs avant / après
        error_orig = straightness_error(row0_orig)
        error_und = straightness_error(row0_und)

    # Retourne les métriques (ret indique si le test est valide)
    return ret, error_orig, error_und


# -------------------------------------------------------------------
# Paramètres et exécution du test pour deux caméras
# -------------------------------------------------------------------

CHECKERBOARD = (9, 6)

# Dossiers d’images utilisés pour calibrer chaque caméra
path1 = "calib_images/cam1"
path2 = "calib_images/cam2"

# Images test indépendantes (une par caméra)
path_test1 = "path/to/testimg.jpg"
path_test2 = "path/to/testimg2.jpg"

# Test de rectitude cam1 et cam2
ret1, error_orig1, error_und1 = test_error(path1, path_test1, CHECKERBOARD)
ret2, error_orig2, error_und2 = test_error(path2, path_test2, CHECKERBOARD)

# Affichage si les deux damiers ont été détectés
if ret1 and ret2:
    print(f"Straightness error 1 (original): {error_orig1}px")
    print(f"Straightness error 1 (undistorted): {error_und1}px")
    print(f"Straightness error 2 (original): {error_orig2}px")
    print(f"Straightness error 2 (undistorted): {error_und2}px")
