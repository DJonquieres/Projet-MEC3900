import cv2
import numpy as np 
import glob

"""
Fichier : zhang_calibration.py

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-11-18

Description :
Ce script implémente une calibration intrinsèque de caméra basée sur la méthode de Zhang,
à partir d’images d’un damier (chessboard).

Objectif :
- Estimer les paramètres intrinsèques de la caméra (matrice K),
- Estimer les coefficients de distorsion optique (radiale et tangentielle),
- Fournir les paramètres nécessaires à :
  - la correction de distorsion (undistortion),
  - la normalisation des coordonnées image,
  - la calibration stéréo et la triangulation 3D.

Lien avec le pipeline du projet :
- Étape fondamentale avant toute triangulation ou modélisation 3D
- Garantit une géométrie image cohérente et mesurable
"""


def calibration(path, CHECKERBOARD=(9, 6)):
    """
    Calibre une caméra à partir d’un ensemble d’images de damier.

    Paramètres
    ----------
    path : str
        Dossier contenant les images du damier (format .jpg).
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
        Vecteurs de rotation (un par image).
    tvecs : list[np.ndarray]
        Vecteurs de translation (un par image).

    Notes
    -----
    Cette fonction applique directement cv2.calibrateCamera, qui implémente
    la méthode de Zhang pour la calibration intrinsèque.
    """

    # Génération des points 3D du damier dans son propre repère
    # Hypothèse : le damier est plan (Z = 0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)

    # Listes de correspondances :
    # - objpoints : points 3D connus (damier)
    # - imgpoints : points 2D détectés dans l’image
    objpoints = []  # Points 3D (monde)
    imgpoints = []  # Points 2D (image)

    # Lecture de toutes les images .jpg dans le dossier
    images = glob.glob(path + "/*.jpg")

    for fname in images:
        # Lecture image
        img = cv2.imread(fname)

        # Conversion en niveaux de gris (requis pour la détection du damier)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détection des coins du damier
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            # Si le damier est détecté :
            # - on ajoute les points 3D correspondants
            # - on raffine les coins détectés en subpixel
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                )
            )
            imgpoints.append(corners2)

    # Calibration intrinsèque OpenCV :
    # Estime K, dist, rvecs et tvecs à partir des correspondances 3D–2D
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    return (ret, K, dist, rvecs, tvecs)


if __name__ == '__main__':
    """
    Exécution directe du script pour calibrer deux caméras (cam1 et cam2).

    Étapes :
    1) Chargement des images de calibration pour chaque caméra,
    2) Estimation des paramètres intrinsèques,
    3) Affichage des matrices K et des coefficients de distorsion.

    Remarque :
    Les paramètres affichés sont ensuite réutilisés dans :
    - la correction de distorsion,
    - la calibration stéréo,
    - la triangulation 3D.
    """

    CHECKERBOARD = (9, 6)

    # Dossiers contenant les images de calibration intrinsèque
    path1 = "calib_images/intrinsics/cam1"
    path2 = "calib_images/intrinsics/cam2"

    # Calibration caméra 1
    ret1, K1, dist1, rvecs1, tvecs1 = calibration(path1, CHECKERBOARD)

    # Calibration caméra 2
    ret2, K2, dist2, rvecs2, tvecs2 = calibration(path2, CHECKERBOARD)

    # Affichage des résultats
    print('Intrinsic matrix K1:')
    print(K1)
    print('Intrinsic matrix K2:')
    print(K2)

    print('\nDistortion coefficients dist1 (k1, k2, p1, p2, k3):')
    print(dist1.ravel())
    print('\nDistortion coefficients dist2 (k1, k2, p1, p2, k3):')
    print(dist2.ravel())