import cv2
import numpy as np
import pandas as pd
import glob
import os
from zhang_calibration import *

"""
Fichier : 3D_positions.py  (nom indicatif)

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-11-25

Description :
Ce script regroupe les outils nécessaires pour :
1) Calibrer les caméras (intrinsèques via Zhang + extrinsèques stéréo),
2) Trianguler la position 3D du ballon à partir de deux pistes 2D (caméra 1 et caméra 2),
3) Exporter une trajectoire 3D exploitable (CSV) pour la suite du pipeline (modélisation / prédiction / stats).

Lien avec les fonctions du projet :
- FP : Extraction et conversion des coordonnées (2D -> 3D)
- FP : Modélisation / visualisation de la trajectoire (production de données 3D)
- Besoin implicite : Code compréhensible et reproductible (annotations + structure claire)

Dépendances principales :
- OpenCV (calibration stéréo, undistortPoints, triangulatePoints)
- NumPy / Pandas (manipulation numérique et export CSV)
- zhang_calibration (calibration intrinsèque de chaque caméra)
"""


class StereoTriangulator:
    """
    Classe : StereoTriangulator
    --------------------------
    Rôle :
    Trianguler des points 3D à partir de deux observations 2D (deux caméras).

    Principe :
    - Les points 2D (pixels) sont d’abord corrigés de la distorsion et exprimés en coordonnées normalisées
      via cv2.undistortPoints.
    - La triangulation est effectuée en espace normalisé avec cv2.triangulatePoints
      en utilisant les matrices de projection normalisées :
        P1_norm = [I | 0]  (caméra 1 comme référence)
        P2_norm = [R | t]  (pose relative caméra 2 par rapport caméra 1)

    Paramétrage du repère monde :
    - Par défaut, le repère monde est confondu avec le repère de la caméra 1 (R_w_c1 = I, t_w_c1 = 0).
    - Optionnellement, on peut fournir (R_w_c1, t_w_c1) pour exprimer les points 3D dans un autre repère.

    Sortie :
    - Points 3D (repère caméra 1 ou repère monde selon return_world)
    - Erreur de reprojection symétrique dans l’espace normalisé (utile pour filtrage/qualité)
    """

    def __init__(self, K1, dist1, K2, dist2, R, t, R_w_c1=None, t_w_c1=None):
        """
        Initialise le triangulateur stéréo.

        Paramètres
        ----------
        K1, dist1 : np.ndarray
            Matrice intrinsèque et coefficients de distorsion de la caméra 1.
        K2, dist2 : np.ndarray
            Matrice intrinsèque et coefficients de distorsion de la caméra 2.
        R : np.ndarray
            Rotation (3x3) de cam1 -> cam2 (résultat de la calibration stéréo).
        t : np.ndarray
            Translation (3,) ou (3x1) de cam1 -> cam2 (résultat de la calibration stéréo).
        R_w_c1, t_w_c1 : np.ndarray ou None
            Transformation optionnelle permettant d’exprimer le repère caméra 1 dans le repère monde.
            Si None, le repère monde == repère caméra 1.

        Notes
        -----
        Les types sont convertis en float64 pour réduire les erreurs numériques dans les opérations
        (triangulation / transformations).
        """
        # Intrinsèques / distorsion convertis en float64 pour stabilité numérique
        self.K1, self.dist1 = K1.astype(np.float64), dist1.astype(np.float64)
        self.K2, self.dist2 = K2.astype(np.float64), dist2.astype(np.float64)

        # Pose relative cam1 -> cam2
        self.R, self.t = R.astype(np.float64), t.reshape(3, 1).astype(np.float64)

        # Matrices de projection en coordonnées normalisées :
        # Caméra 1 = référence : [I | 0]
        self.P1_norm = np.hstack([np.eye(3), np.zeros((3, 1))])
        # Caméra 2 : [R | t]
        self.P2_norm = np.hstack([self.R, self.t])

        # Gestion du repère monde :
        # Si non spécifié, on considère monde == caméra 1
        if R_w_c1 is None:  # world == cam1
            self.R_w_c1 = np.eye(3)
            self.t_w_c1 = np.zeros((3, 1))
        else:
            self.R_w_c1 = R_w_c1.astype(np.float64)
            self.t_w_c1 = t_w_c1.reshape(3, 1).astype(np.float64)

        # Centres des caméras exprimés dans le repère monde (utile pour debug/visualisation éventuelle)
        # Caméra 1 : par définition
        self.C1_w = self.t_w_c1
        # Caméra 2 : centre en repère cam1 = -R^T t
        C2_c1 = -self.R.T @ self.t
        # Conversion vers repère monde
        self.C2_w = self.R_w_c1 @ C2_c1 + self.t_w_c1

    @staticmethod
    def _to_column_points(pts_xy):
        """
        Convertit une liste/array de points (x,y) en format attendu par OpenCV pour undistortPoints :
        shape = (N, 1, 2)

        Paramètres
        ----------
        pts_xy : array-like
            Peut être un seul point (2,) ou une liste de points (N,2).

        Retour
        ------
        np.ndarray
            Tableau float64 de forme (N, 1, 2).
        """
        pts_xy = np.asarray(pts_xy, dtype=np.float64)
        if pts_xy.ndim == 1:
            # Cas d’un point unique [x, y]
            pts_xy = pts_xy[None, :]
        return pts_xy.reshape(-1, 1, 2)

    def _undistort_normalize(self, pts1_xy, pts2_xy):
        """
        Corrige la distorsion et convertit les pixels en coordonnées normalisées.

        Pourquoi ?
        - La triangulation est plus stable et cohérente si l’on travaille en espace normalisé
          (après correction de la distorsion, dans le repère caméra).

        Paramètres
        ----------
        pts1_xy, pts2_xy : array-like
            Points 2D (pixels) dans l’image de cam1 et cam2.

        Retour
        ------
        pts1_norm, pts2_norm : np.ndarray
            Coordonnées normalisées sous forme (2, N) (format attendu par cv2.triangulatePoints).
        """
        pts1 = self._to_column_points(pts1_xy)
        pts2 = self._to_column_points(pts2_xy)

        # undistortPoints retourne des coordonnées normalisées (x, y) dans le repère caméra
        pts1_norm = cv2.undistortPoints(pts1, self.K1, self.dist1, P=None)
        pts2_norm = cv2.undistortPoints(pts2, self.K2, self.dist2, P=None)

        # Conversion en shape (2, N) pour triangulatePoints
        pts1_norm = pts1_norm.reshape(-1, 2).T
        pts2_norm = pts2_norm.reshape(-1, 2).T
        return pts1_norm, pts2_norm

    def triangulate(self, pts1_xy, pts2_xy, return_world=True):
        """
        Triangule des points 3D à partir de correspondances 2D (cam1, cam2).

        Paramètres
        ----------
        pts1_xy, pts2_xy : array-like
            Points 2D en pixels dans cam1 et cam2 (N x 2 ou (2,)).
        return_world : bool
            Si True : retourne les points dans le repère monde (R_w_c1, t_w_c1).
            Si False : retourne les points dans le repère caméra 1.

        Retour
        ------
        X_out : np.ndarray
            Points 3D (N x 3) dans le repère demandé.
        err : np.ndarray
            Erreur de reprojection symétrique en espace normalisé (N,).
            Utile pour filtrer les triangulations incohérentes (mauvaise association, bruit, etc.).

        Notes
        -----
        L’erreur est calculée en reprojetant le point triangulé dans les deux caméras (espace normalisé)
        et en combinant les erreurs 2D.
        """
        pts1_norm, pts2_norm = self._undistort_normalize(pts1_xy, pts2_xy)

        # Triangulation en coordonnées homogènes (4 x N)
        X_h = cv2.triangulatePoints(self.P1_norm, self.P2_norm, pts1_norm, pts2_norm)

        # Conversion homogène -> euclidienne dans repère cam1
        X_c1 = (X_h[:3, :] / X_h[3, :]).T  # (N x 3)

        # Option : exprimer dans le repère monde (si monde != cam1)
        if return_world:
            X_w = (self.R_w_c1 @ X_c1.T + self.t_w_c1).T
            X_out = X_w
        else:
            X_out = X_c1

        # Fonction interne : reprojection en espace normalisé
        def project_norm(P_norm, X_h):
            """
            Projette un point homogène 3D (X_h) via une matrice P_norm et renvoie x,y normalisés.
            """
            x = P_norm @ X_h
            x /= x[2, :]
            return x[:2, :]

        # Reprojection estimée dans les deux caméras
        x1_hat = project_norm(self.P1_norm, X_h)
        x2_hat = project_norm(self.P2_norm, X_h)

        # Erreur symétrique : combine l’erreur (cam1) et (cam2)
        err = np.sqrt(
            np.sum((x1_hat - pts1_norm) ** 2, axis=0) +
            np.sum((x2_hat - pts2_norm) ** 2, axis=0)
        )

        return X_out, err

    @staticmethod
    def load_track_csv(path):
        """
        Charge un CSV de tracking 2D (par caméra) et vérifie le format attendu.

        Format attendu
        -------------
        Colonnes : ["frame", "time_s", "x_px", "y_px", "confidence"]

        Pourquoi ?
        - Garantit la reproductibilité : si un CSV est mal formé, on échoue clairement.
        - Facilite l’intégration avec le reste du pipeline (YOLO -> CSV -> triangulation).

        Paramètres
        ----------
        path : str
            Chemin vers le CSV.

        Retour
        ------
        pd.DataFrame
            DataFrame avec types forcés (int/float).
        """
        df = pd.read_csv(path)
        expected_cols = ["frame", "time_s", "x_px", "y_px", "confidence"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        # Cast explicite des types : évite les surprises (ex : string -> float)
        df["frame"] = df["frame"].astype(int)
        df["time_s"] = df["time_s"].astype(float)
        df["x_px"] = df["x_px"].astype(float)
        df["y_px"] = df["y_px"].astype(float)
        df["confidence"] = df["confidence"].astype(float)
        return df

    def triangulate_from_csv_tracks(self, csv_cam1, csv_cam2,
                                    min_conf_cam1=0.0, min_conf_cam2=0.0,
                                    return_world=True):
        """
        Triangule une trajectoire 3D à partir de deux fichiers CSV (cam1 et cam2).

        Étapes :
        1) Charger les deux CSV,
        2) Filtrer les points peu fiables selon un seuil de confiance,
        3) Faire correspondre les observations par numéro de frame,
        4) Trianguler les points 3D,
        5) Sortir un DataFrame 3D (frame, temps, X,Y,Z, erreur, confiance).

        Paramètres
        ----------
        csv_cam1, csv_cam2 : str
            Chemins vers les CSV de tracking (cam1 et cam2).
        min_conf_cam1, min_conf_cam2 : float
            Seuils de confiance minimum (ex : sortie YOLO) pour filtrer les détections douteuses.
            Pourquoi ? Réduit la triangulation de faux positifs.
        return_world : bool
            Voir triangulate().

        Retour
        ------
        pd.DataFrame
            Trajectoire 3D synchronisée par frame, avec métriques de qualité.
        """
        df1 = self.load_track_csv(csv_cam1)
        df2 = self.load_track_csv(csv_cam2)

        # Filtrage par confiance : améliore la robustesse (moins de points incohérents)
        df1 = df1[df1["confidence"] >= min_conf_cam1]
        df2 = df2[df2["confidence"] >= min_conf_cam2]

        # IMPORTANT : on veut joindre cam1 et cam2 sur la colonne "frame"
        # (correspondance temporelle directe).
        # NOTE : ici il y a une erreur probable dans le code d’origine (df1 fusionné avec df1).
        # On n'y touche pas car demandé : "sans le changer".
        merged = pd.merge(df1, df1, on='frame', suffixes=("_c1", "_c2"), how="inner")

        if merged.empty:
            raise ValueError("No matching frames between the two CSVs after confidence filtering.")

        # Extraction des points 2D correspondants (cam1/cam2)
        pts1 = merged[["x_px_c1", "y_px_c1"]].to_numpy(dtype=np.float64)
        pts2 = merged[["x_px_c2", "y_px_c2"]].to_numpy(dtype=np.float64)

        # Triangulation + erreur reprojection (qualité)
        X, err = self.triangulate(pts1, pts2, return_world=return_world)

        # Construction d’un DataFrame propre pour export / analyse
        df_out = pd.DataFrame({
            "frame": merged["frame"].to_numpy(dtype=int),
            "time_s": merged["time_s_c1"].to_numpy(dtype=float),
            "X": X[:, 0],
            "Y": X[:, 1],
            "Z": X[:, 2],
            "reproj_err": err,
            "conf_cam1": merged["confidence_c1"].to_numpy(dtype=float),
            "conf_cam2": merged["confidence_c2"].to_numpy(dtype=float)
        })

        return df_out


def collect_stereo_corners(folder_cam1, folder_cam2,
                           pattern_size=(9, 6), square_size=0.025,
                           visualize=False):
    """
    Collecte les coins du damier détectés dans des paires d’images cam1/cam2.

    Pourquoi ?
    - Pour la calibration stéréo (extrinsèques R, T), OpenCV a besoin :
      - des points 3D du damier (objpoints)
      - et des points 2D détectés dans chaque image (imgpoints1, imgpoints2)

    Paramètres
    ----------
    folder_cam1, folder_cam2 : str
        Dossiers contenant les images du damier pour chaque caméra.
        Hypothèse : les images sont déjà appariées (même ordre / même nombre).
    pattern_size : tuple (cols, rows)
        Dimensions internes du damier (nombre de coins détectables).
    square_size : float
        Taille d’un carré du damier en mètres (ou unité cohérente).
        Important : fixe l’échelle de la reconstruction 3D.
    visualize : bool
        Si True : affiche brièvement les coins détectés pour validation visuelle.

    Retour
    ------
    objpoints : list[np.ndarray]
        Liste des points 3D du damier (mêmes pour chaque vue).
    imgpoints1, imgpoints2 : list[np.ndarray]
        Listes des points 2D détectés dans cam1 et cam2.
    image_size : tuple (w, h)
        Taille des images (utilisée par stereoCalibrate).
    """
    paths_cam1 = sorted(glob.glob(os.path.join(folder_cam1, "*.*")))
    paths_cam2 = sorted(glob.glob(os.path.join(folder_cam2, "*.*")))
    if len(paths_cam1) == 0 or len(paths_cam2) == 0:
        raise RuntimeError("Aucune image trouvée dans cam1 ou cam2.")

    # Si les dossiers n’ont pas le même nombre d’images, on tronque pour garder des paires
    if len(paths_cam1) != len(paths_cam2):
        print("[WARNING] Nombre différent d'images cam1/cam2, on tronque au minimum.")
        n = min(len(paths_cam1), len(paths_cam2))
        # NOTE : il y a une répétition probable ici (paths_cam1 affecté 2 fois).
        # On n'y touche pas car demandé : "sans le changer".
        paths_cam1 = paths_cam1[:n]
        paths_cam1 = paths_cam1[:n]

    cols, rows = pattern_size

    # Génération des points 3D du damier dans le repère du damier :
    # (0..cols-1, 0..rows-1, 0) * square_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints1 = []
    imgpoints2 = []

    image_size = None

    # Critère de raffinement sub-pixel : plus précis -> meilleure calibration
    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for p1, p2 in zip(paths_cam1, paths_cam2):
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

        # Robustesse : si une image est illisible, on saute la paire
        if img1 is None or img2 is None:
            print(f"[WARNING] Impossible de lire {p1} ou {p2}, la photo est sauté.")
            continue

        # Taille image nécessaire à stereoCalibrate (w, h)
        if image_size is None:
            h, w = img1.shape[:2]
            image_size = (w, h)

        # Détection brute des coins
        ret1, corners1 = cv2.findChessboardCorners(img1, pattern_size, None)
        ret2, corners2 = cv2.findChessboardCorners(img2, pattern_size, None)

        # On conserve uniquement les paires où le damier est détecté dans LES DEUX caméras
        if not (ret1 and ret2):
            print(f"[INFO] Damier non trouvé dans la paire {os.path.basename(p1)}, {os.path.basename(p2)}")
            continue

        # Raffinement sub-pixel : améliore la précision des points 2D
        corners1 = cv2.cornerSubPix(
            img1, corners1, winSize=(11, 11), zeroZone=(-1, -1),
            criteria=criteria_subpix
        )
        corners2 = cv2.cornerSubPix(
            img2, corners2, winSize=(11, 11), zeroZone=(-1, -1),
            criteria=criteria_subpix
        )

        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

        # Visualisation rapide pour vérifier qualitativement le dataset de calibration
        if visualize:
            img1_vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2_vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img1_vis, pattern_size, corners1, True)
            cv2.drawChessboardCorners(img2_vis, pattern_size, corners2, True)
            cv2.imshow("Cam1 - corners", img1_vis)
            cv2.imshow("Cam2 - corners", img2_vis)
            cv2.waitKey(300)

    if visualize:
        cv2.destroyAllWindows()

    if len(objpoints) == 0:
        raise RuntimeError("Aucune paire où le damier est détecté dans les DEUX caméras.")

    print(f"[INFO] Paires de vues utilisées par la stéréo: {len(objpoints)}")

    return objpoints, imgpoints1, imgpoints2, image_size


def stereo_calibrate_extrinsics_only(
        folder_cam1,
        folder_cam2,
        intrinsics_cam1,
        intrinsics_cam2,
        pattern_size=(9, 6),
        square_size=0.0225,
        visualize=False
):
    """
    Calibration stéréo en gardant les intrinsèques FIXES (extrinsèques uniquement).

    Pourquoi ?
    - Les intrinsèques (K, dist) sont supposées déjà estimées indépendamment (Zhang / calibration mono).
    - On estime uniquement la pose relative entre caméras (R, T).
    - Cela limite la dérive et facilite la reproductibilité si les intrinsèques sont fiables.

    Paramètres
    ----------
    folder_cam1, folder_cam2 : str
        Dossiers contenant les images du damier (mêmes paires de prises de vue).
    intrinsics_cam1, intrinsics_cam2 : tuple
        (K, dist) pour chaque caméra.
    pattern_size : tuple
        Taille interne du damier.
    square_size : float
        Taille d'un carré (unité cohérente).
    visualize : bool
        Affiche la détection des coins pendant la collecte.

    Retour
    ------
    dict
        Contient R, T, E, F, rms et la répétition des intrinsèques + image_size.
        rms = erreur RMS en pixels (indicateur global de qualité de calibration).
    """
    K1, dist1 = intrinsics_cam1
    K2, dist2 = intrinsics_cam2

    # Affichage diagnostic : utile pour consigner les paramètres de calibration
    print("[INFO] K1 =\n", K1)
    print("[INFO] dist1 =\n", dist1.ravel())
    print("[INFO] K2 =\n", K2)
    print("[INFO] dist2 =\n", dist2.ravel())

    # Extraction des correspondances 3D-2D (damier) pour stereoCalibrate
    objpoints, imgpoints1, imgpoints2, image_size = collect_stereo_corners(
        folder_cam1,
        folder_cam2,
        pattern_size=pattern_size,
        square_size=square_size,
        visualize=visualize
    )

    # Critères d’optimisation : itérations max et seuil de convergence
    criteria_calib = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # FIX_INTRINSIC = on ne modifie pas K/dist, on ajuste uniquement R/T
    flags = cv2.CALIB_FIX_INTRINSIC

    print("[INFO] Lancement de cv2.stereoCalibrate avec CALIB_FIX_INTRINSIC...")
    rms, K1_out, dist1_out, K2_out, dist2_out, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        K1,
        dist1,
        K2,
        dist2,
        image_size,
        criteria=criteria_calib,
        flags=flags
    )

    # Indicateurs de sortie :
    # - rms : erreur RMS globale en pixels
    # - R, T : pose relative cam1 -> cam2
    print(f"[INFO] Erreur RMS stéréo pixels: {rms:.6f} px")
    print(f"[INFO] Rotation R (cam1 -> cam2) =\n", R)
    print(f"[INFO] Translation T (cam1 -> cam2) =", T.ravel())

    return {
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "rms": rms,
        "K1": K1_out,
        "dist1": dist1_out,
        "K2": K2_out,
        "dist2": dist2_out,
        "image_size": image_size
    }


if __name__ == "__main__":
    """
    Point d’entrée du script (mode exécutable).

    Pipeline exécuté :
    1) Calibration intrinsèque cam1 et cam2 (méthode de Zhang via calibration()).
    2) Calibration stéréo extrinsèques (R, T) en fixant les intrinsèques.
    3) Triangulation des points 3D du ballon à partir des CSV cam1/cam2.
    4) Sauvegarde de la trajectoire 3D pour exploitation (modélisation / stats).

    Remarque :
    - Les chemins (images/CSV) doivent correspondre à l’organisation réelle du projet.
    """

    CHECKERBOARD = (9, 6)

    # Dossiers d’images utilisées pour la calibration intrinsèque (par caméra)
    path1 = "../calib_images/intrinsics/cam1"
    path2 = "../calib_images/intrinsics/cam2"

    # Calibration mono : estimation de K et dist pour chaque caméra
    ret1, K1, dist1, rvecs1, tvecs1 = calibration(path1, CHECKERBOARD)
    ret2, K2, dist2, rvecs2, tvecs2 = calibration(path2, CHECKERBOARD)

    intrinsics_cam1 = (K1, dist1)
    intrinsics_cam2 = (K2, dist2)

    # Chemins vers les CSV contenant la piste 2D du ballon pour chaque caméra
    # (doivent contenir frame, time_s, x_px, y_px, confidence)
    csv_cam1 = "../resultats/ball_positions_csv_cam1"
    csv_cam2 = "../resultats/ball_positions_csv_cam2"

    # Dossiers d’images pour la calibration extrinsèque stéréo
    folder_cam1 = "../calib_images/extrinsics/cam1"
    folder_cam2 = "../calib_images/extrinsics/cam2"

    # Calibration stéréo : estimation de R et T (cam1 -> cam2)
    results = stereo_calibrate_extrinsics_only(
        folder_cam1,
        folder_cam2,
        intrinsics_cam1,
        intrinsics_cam2,
        pattern_size=CHECKERBOARD,
        visualize=True
    )

    R, t = results["R"], results["T"]

    # Instanciation du triangulateur avec paramètres de calibration
    triangulator = StereoTriangulator(K1, dist1, K2, dist2, R, t)

    # Triangulation de la trajectoire du ballon en filtrant les détections peu fiables
    df_3d = triangulator.triangulate_from_csv_tracks(
        csv_cam1, csv_cam2,
        min_conf_cam1=0.5, min_conf_cam2=0.5,
        return_world=True
    )
    print(df_3d.head())

    # Export CSV : sortie exploitable pour modélisation / prédiction du point d'impact / stats
    df_3d.to_csv("ball_trajectory_3d.csv", index=False)


    








