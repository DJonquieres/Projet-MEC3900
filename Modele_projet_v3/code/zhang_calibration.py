import cv2
import numpy as np 
import glob

"""
Fichier : zhang_calibration.py

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-11-27

Description :
Ce script effectue une calibration intrinsèque de caméra à partir d’images d’un damier
(chessboard), selon la méthode de Zhang (implémentée via OpenCV).

Objectif :
- Estimer la matrice intrinsèque K (paramètres internes de la caméra),
- Estimer les coefficients de distorsion (k1, k2, p1, p2, k3),
- Produire les paramètres nécessaires pour :
  - corriger la distorsion (undistort),
  - normaliser les coordonnées image,
  - calibrer en stéréo et trianguler en 3D.

Lien avec le pipeline du projet :
- Étape fondamentale avant la reconstruction 3D : une mauvaise calibration intrinsèque
  entraîne des erreurs importantes lors de la triangulation.
"""


def calibration(path, CHECKERBOARD=(9, 6)):
    """
    Calibre une caméra à partir d’un dossier d’images de damier.

    Paramètres
    ----------
    path : str
        Dossier contenant les images de calibration (format .jpg).
    CHECKERBOARD : tuple (cols, rows)
        Dimensions internes du damier (nombre de coins détectables).

    Retour
    ------
    ret : float
        Erreur RMS de reprojection globale (en pixels).
    K : np.ndarray (3x3)
        Matrice intrinsèque de la caméra.
    dist : np.ndarray
        Coefficients de distorsion (k1, k2, p1, p2, k3).
    rvecs : list[np.ndarray]
        Rotations estimées (une par image).
    tvecs : list[np.ndarray]
        Translations estimées (une par image).
    """

    # Création des points 3D du damier dans le repère damier (Z=0)
    # Remarque : square_size n’est pas utilisé ici, donc l’échelle reste arbitraire,
    # ce qui est suffisant pour estimer K et dist.
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Listes des correspondances 3D-2D nécessaires à calibrateCamera
    objpoints = []  # Points 3D (damier)
    imgpoints = []  # Points 2D (coins détectés dans l’image)

    # Lecture de toutes les images dans le dossier
    images = glob.glob(path + "/*.jpg")

    for fname in images:
        # Lecture image
        img = cv2.imread(fname)

        # Conversion en niveaux de gris (requis pour findChessboardCorners)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détection des coins du damier
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            # Ajout des points 3D correspondants (mêmes pour chaque image)
            objpoints.append(objp)

            # Raffinement subpixel : améliore la précision des coins -> meilleure calibration
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            # Ajout des points 2D raffinés
            imgpoints.append(corners2)

    # Calibration intrinsèque OpenCV :
    # Estime K, dist et les poses (rvecs, tvecs) qui expliquent les observations
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return (ret, K, dist, rvecs, tvecs)


if __name__ == '__main__':
    """
    Exécution directe : calibration de deux caméras (cam1 et cam2) et affichage des résultats.

    Étapes :
    1) Charger les images de calibration intrinsèque de cam1 et cam2,
    2) Estimer K et dist pour chaque caméra,
    3) Afficher K1/K2 et dist1/dist2.

    Remarque :
    Les paramètres affichés sont ensuite réutilisés dans la calibration stéréo
    et la triangulation 3D.
    """

    CHECKERBOARD = (9, 6)

    # Dossiers contenant les images de calibration intrinsèque
    path1 = "../calib_images/intrinsics/cam1"
    path2 = "../calib_images/intrinsics/cam2"

    # Calibration caméra 1
    ret1, K1, dist1, rvecs1, tvecs1 = calibration(path1, CHECKERBOARD)

    # Calibration caméra 2
    ret2, K2, dist2, rvecs2, tvecs2 = calibration(path2, CHECKERBOARD)

    # Affichage des matrices intrinsèques
    print('Intrinsic matrix K1:')
    print(K1)
    print('Intrinsic matrix K2:')
    print(K2)

    # Affichage des coefficients de distorsion
    # (k1, k2 : distorsion radiale, p1, p2 : distorsion tangentielle, k3 : radiale)
    print('n/Distortion coefficients dist1 (k1, k2, p1, p2, k3):')
    print(dist1.ravel())
    print('n/Distortion coefficients dist2 (k1, k2, p1, p2, k3):')
    print(dist2.ravel())

# -------------------------------------------------------------------
# Section optionnelle (commentée) : exemple d’undistortion d’une image test
# -------------------------------------------------------------------
# Objectif :
# - Visualiser qualitativement la correction de distorsion sur une image.
#
# img = cv2.imread("path/to/testimg.jpg")
# img2 = cv2.imread("path/to/testimg2.jpg")
# h,w = img.shape[:2]
# new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w,h))
#
# undistorted = cv2.undistort(img, K, dist, None, new_K)
# cv2.imwrite("undistorted2.jpg", undistorted)

