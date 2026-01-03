import cv2
import numpy as np
import pandas as pd
import glob
import os
from zhang_calibration import *

"""
Fichier : 3D_positions.py

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-11-27

Description :
Ce script regroupe les outils nécessaires pour :
1) Calibrer chaque caméra (intrinsèques K, dist) via la méthode de Zhang,
2) Estimer la pose relative entre deux caméras (extrinsèques stéréo : R, T),
3) Trianguler la position 3D du ballon à partir de deux pistes 2D (CSV cam1 et cam2),
4) Exporter un fichier CSV contenant la trajectoire 3D (X, Y, Z) et une métrique de qualité
   (erreur de reprojection).

Lien avec le pipeline du projet :
- Extraction et conversion des coordonnées (2D -> 3D)
- Base pour la modélisation de trajectoire et la prédiction du point d’impact
- Robustesse : filtrage par confiance + calcul d’erreur de reprojection
"""


class StereoTriangulator:
    """" 
    Classe : StereoTriangulator
    --------------------------
    Triangule des points 3D à partir de deux caméras.

    Principe :
    - Correction de distorsion + passage en coordonnées normalisées (undistortPoints),
    - Triangulation en espace normalisé (cv2.triangulatePoints) :
        P1_norm = [I | 0]  (caméra 1 utilisée comme référence)
        P2_norm = [R | t]  (pose relative caméra 2 par rapport caméra 1)
    - Option : conversion vers un repère monde défini par (R_w_c1, t_w_c1).
    """
    def __init__(self, K1, dist1, K2, dist2, R, t, R_w_c1=None, t_w_c1=None):
        """
        Initialise le triangulateur avec :
        - Intrinsèques + distorsion de chaque caméra,
        - Extrinsèques stéréo (R, t) cam1 -> cam2,
        - Optionnel : transformation repère caméra 1 -> repère monde.
        """
        self.K1, self.dist1 = K1.astype(np.float64), dist1.astype(np.float64)
        self.K2, self.dist2 = K2.astype(np.float64), dist2.astype(np.float64)
        self.R, self.t = R.astype(np.float64), t.reshape(3, 1).astype(np.float64)

        # Matrices de projection en coordonnées normalisées :
        # cam1 = [I|0], cam2 = [R|t]
        self.P1_norm = np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2_norm = np.hstack([self.R, self.t])

        # Par défaut, on prend repère monde == repère caméra 1
        if R_w_c1 is None:  # world == cam1
            self.R_w_c1 = np.eye(3)
            self.t_w_c1 = np.zeros((3, 1))
        else:
            self.R_w_c1 = R_w_c1.astype(np.float64)
            self.t_w_c1 = t_w_c1.reshape(3, 1).astype(np.float64)

        # Centres caméra (utile debug / visualisation / vérification géométrique)
        self.C1_w = self.t_w_c1
        C2_c1 = -self.R.T @ self.t
        self.C2_w = self.R_w_c1 @ C2_c1 + self.t_w_c1

    @staticmethod
    def _to_column_points(pts_xy):
        """
        Met les points au format OpenCV (N,1,2) attendu par undistortPoints.
        Accepte un point unique (2,) ou une liste de points (N,2).
        """
        pts_xy = np.asarray(pts_xy, dtype=np.float64)
        if pts_xy.ndim == 1:
            pts_xy = pts_xy[None, :]
        return pts_xy.reshape(-1, 1, 2)

    def _undistort_normalize(self, pts1_xy, pts2_xy):
        """
        Corrige la distorsion et renvoie des coordonnées normalisées (2, N),
        adaptées à cv2.triangulatePoints.
        """
        pts1 = self._to_column_points(pts1_xy)
        pts2 = self._to_column_points(pts2_xy)

        # P=None -> coordonnées normalisées (repère caméra) après correction de distorsion
        pts1_norm = cv2.undistortPoints(pts1, self.K1, self.dist1, P=None)
        pts2_norm = cv2.undistortPoints(pts2, self.K2, self.dist2, P=None)

        pts1_norm = pts1_norm.reshape(-1, 2).T
        pts2_norm = pts2_norm.reshape(-1, 2).T
        return pts1_norm, pts2_norm

    def triangulate(self, pts1_xy, pts2_xy, return_world=True):
        """
        Triangule des points 3D à partir des correspondances 2D (cam1/cam2).

        Retour :
        - X_out (N,3) : points 3D (repère cam1 ou monde),
        - err (N,) : erreur de reprojection symétrique en espace normalisé
                    (indicateur de qualité).
        """
        pts1_norm, pts2_norm = self._undistort_normalize(pts1_xy, pts2_xy)

        # Triangulation homogène (4 x N)
        X_h = cv2.triangulatePoints(self.P1_norm, self.P2_norm, pts1_norm, pts2_norm)

        # Passage homogène -> euclidien (repère cam1)
        X_c1 = (X_h[:3, :] / X_h[3, :]).T

        # Conversion optionnelle vers repère monde
        if return_world:
            X_w = (self.R_w_c1 @ X_c1.T + self.t_w_c1).T
            X_out = X_w
        else:
            X_out = X_c1

        # Reprojection en espace normalisé pour calcul d’erreur (qualité)
        def project_norm(P_norm, X_h):
            # Erreur épipolaire/reprojection dans l’espace normalisé
            x = P_norm @ X_h
            x /= x[2, :]
            return x[:2, :]

        x1_hat = project_norm(self.P1_norm, X_h)
        x2_hat = project_norm(self.P2_norm, X_h)

        # Erreur symétrique combinée (cam1 + cam2)
        err = np.sqrt(
            np.sum((x1_hat - pts1_norm) ** 2, axis=0) +
            np.sum((x2_hat - pts2_norm) ** 2, axis=0)
        )

        return X_out, err

    @staticmethod
    def load_track_csv(path):
        """
        Charge un CSV de tracking (sortie YOLO) et valide le format attendu.

        Colonnes attendues :
        ["frame", "time_s", "x_px", "y_px", "confidence"]
        """
        df = pd.read_csv(path)
        expected_cols = ["frame", "time_s", "x_px", "y_px", "confidence"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        # Cast explicite : évite erreurs silencieuses (string -> float, etc.)
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
        Triangule une trajectoire 3D à partir de deux CSV (cam1 et cam2).

        Étapes :
        1) Charger CSV cam1/cam2,
        2) Filtrer par confiance (réduction des faux positifs),
        3) Garder uniquement les frames communes (synchronisation),
        4) Fusionner et trianguler,
        5) Retourner un DataFrame 3D + métriques de qualité.
        """
        df1 = self.load_track_csv(csv_cam1)
        df2 = self.load_track_csv(csv_cam2)

        # Filtrage par confiance : améliore robustesse de la triangulation
        df1 = df1[df1["confidence"] >= min_conf_cam1]
        df2 = df2[df2["confidence"] >= min_conf_cam2]

        # Conservation des frames communes (assure correspondance temporelle)
        df1 = df1[df1["frame"].isin(df2["frame"])]
        df2 = df2[df2["frame"].isin(df1["frame"])]

        # Fusion par frame pour aligner les observations cam1/cam2
        # NOTE : le code fusionne df1 avec df1 (probable erreur logique),
        # mais on ne le modifie pas car demandé : "sans changer le code".
        merged = pd.merge(df1, df1, on='frame', suffixes=("_c1", "_c2"), how="inner")

        if merged.empty:
            raise ValueError("No matching frames between the two CSVs after confidence filtering.")

        pts1 = merged[["x_px_c1", "y_px_c1"]].to_numpy(dtype=np.float64)
        pts2 = merged[["x_px_c2", "y_px_c2"]].to_numpy(dtype=np.float64)

        # Triangulation
        X, err = self.triangulate(pts1, pts2, return_world=return_world)

        # Export structuré : prêt pour analyse / modélisation
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
    Collecte les coins du damier détectés sur des paires d’images cam1/cam2.

    Pourquoi ?
    - cv2.stereoCalibrate nécessite les correspondances :
      - points 3D connus du damier (objpoints),
      - projections 2D observées dans cam1 et cam2 (imgpoints1, imgpoints2).
    """
    paths_cam1 = sorted(glob.glob(os.path.join(folder_cam1, "*.*")))
    paths_cam2 = sorted(glob.glob(os.path.join(folder_cam2, "*.*")))

    if len(paths_cam1) == 0 or len(paths_cam2) == 0:
        raise RuntimeError("Aucune image trouvée dans cam1 ou cam2.")

    # Si nombre différent, on tronque au minimum pour garder des paires
    if len(paths_cam1) != len(paths_cam2):
        print("[WARNING] Nombre différent d'images cam1/cam2, on tronque au minimum.")
        n = min(len(paths_cam1), len(paths_cam2))
        # NOTE : duplication probable (paths_cam1 affecté deux fois), conservée telle quelle.
        paths_cam1 = paths_cam1[:n]
        paths_cam1 = paths_cam1[:n]

    cols, rows = pattern_size

    # Points 3D du damier (Z=0), échelle donnée par square_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints1 = []
    imgpoints2 = []
    image_size = None

    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for p1, p2 in zip(paths_cam1, paths_cam2):
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

        # Robustesse : ignorer les fichiers illisibles
        if img1 is None or img2 is None:
            print(f"[WARNING] Impossible de lire {p1} ou {p2}, la photo est sauté.")
            continue

        if image_size is None:
            h, w = img1.shape[:2]
            image_size = (w, h)

        # Détection coins damier
        ret1, corners1 = cv2.findChessboardCorners(img1, pattern_size, None)
        ret2, corners2 = cv2.findChessboardCorners(img2, pattern_size, None)

        # On conserve uniquement les paires où les deux caméras détectent le damier
        if not (ret1 and ret2):
            print(f"[INFO] Damier non trouvé dans la paire {os.path.basename(p1)}, {os.path.basename(p2)}")
            continue

        # Raffinement subpixel (meilleure précision -> meilleure calibration)
        corners1 = cv2.cornerSubPix(img1, corners1, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria_subpix)
        corners2 = cv2.cornerSubPix(img2, corners2, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria_subpix)

        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

        # Option : contrôle visuel rapide des coins détectés
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
    Estime uniquement les extrinsèques stéréo (R, T) en fixant les intrinsèques.

    Pourquoi ?
    - Les intrinsèques K/dist sont calibrés indépendamment (plus stable),
    - stereoCalibrate avec CALIB_FIX_INTRINSIC ajuste uniquement la pose relative
      entre caméras, ce qui est nécessaire pour trianguler en 3D.
    """
    K1, dist1 = intrinsics_cam1
    K2, dist2 = intrinsics_cam2

    print("[INFO] K1 =\n", K1)
    print("[INFO] dist1 =\n", dist1.ravel())
    print("[INFO] K2 =\n", K2)
    print("[INFO] dist2 =\n", dist2.ravel())

    objpoints, imgpoints1, imgpoints2, image_size = collect_stereo_corners(
        folder_cam1,
        folder_cam2,
        pattern_size=pattern_size,
        square_size=square_size,
        visualize=visualize
    )

    criteria_calib = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    # FIX_INTRINSIC : on ne ré-optimise pas K/dist
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

    # rms : indicateur global de qualité (reprojection RMS en pixels)
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
    Exécution directe du script.

    Pipeline :
    1) Calibration intrinsèque des deux caméras (K1/dist1 et K2/dist2),
    2) Calibration stéréo extrinsèques (R, T),
    3) Triangulation de la trajectoire du ballon à partir des CSV 2D,
    4) Export CSV 3D.
    """

    CHECKERBOARD = (9, 6)

    # Dossiers images de calibration intrinsèque
    path1 = "../calib_images/intrinsics/cam1"
    path2 = "../calib_images/intrinsics/cam2"

    # Intrinsèques cam1/cam2
    ret1, K1, dist1, rvecs1, tvecs1 = calibration(path1, CHECKERBOARD)
    ret2, K2, dist2, rvecs2, tvecs2 = calibration(path2, CHECKERBOARD)

    intrinsics_cam1 = (K1, dist1)
    intrinsics_cam2 = (K2, dist2)

    # CSV de tracking 2D (sortie YOLO) pour chaque caméra
    csv_cam1 = "../ball_positions_csv_cam1"
    csv_cam2 = "../ball_positions_csv_cam2"

    # Dossiers images de calibration extrinsèque stéréo
    folder_cam1 = "../calib_images/extrinsics/cam1"
    folder_cam2 = "../calib_images/extrinsics/cam2"

    # Estimation de la pose relative entre caméras
    results = stereo_calibrate_extrinsics_only(
        folder_cam1,
        folder_cam2,
        intrinsics_cam1,
        intrinsics_cam2,
        pattern_size=CHECKERBOARD,
        visualize=True
    )

    R, t = results["R"], results["T"]

    # Triangulation 3D
    triangulator = StereoTriangulator(K1, dist1, K2, dist2, R, t)
    df_3d = triangulator.triangulate_from_csv_tracks(
        csv_cam1, csv_cam2,
        min_conf_cam1=0.5, min_conf_cam2=0.5,
        return_world=True
    )
    print(df_3d.head())

    # Export final pour la suite (modélisation / stats)
    df_3d.to_csv("ball_trajectory_3d.csv", index=False)


    








