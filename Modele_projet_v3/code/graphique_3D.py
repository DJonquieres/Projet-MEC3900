import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Fichier : graphique_3D.py

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-11-27

Description :
Ce script permet de visualiser la trajectoire 3D du ballon à partir d’un fichier CSV
contenant les coordonnées reconstruites (X, Y, Z).

Le fichier CSV est typiquement généré par l’étape de triangulation stéréo et constitue
une sortie directe du pipeline de reconstruction 3D.

Lien avec le pipeline du projet :
- Visualisation des résultats de triangulation 3D
- Validation qualitative de la trajectoire reconstruite
- Support à l’analyse et à la présentation des résultats
"""

# -------------------------------------------------------------------
# Chargement des données 3D
# -------------------------------------------------------------------

# Chemin vers le fichier CSV contenant la trajectoire 3D
path = "ball_trajectory_3d.csv"

# Lecture du CSV (colonnes attendues : X, Y, Z)
df = pd.read_csv(path)

# Extraction des coordonnées sous forme de tableaux NumPy
x = df["X"].to_numpy()
y = df["Y"].to_numpy()
z = df["Z"].to_numpy()

# -------------------------------------------------------------------
# Visualisation 3D de la trajectoire
# -------------------------------------------------------------------

# Création de la figure Matplotlib
fig = plt.figure()

# Ajout d’un axe 3D
ax = fig.add_subplot(111, projection="3d")

# Tracé de la trajectoire du ballon
ax.plot(x, y, z, label="Trajectory")

# Annotation des axes (unités en mètres)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# Légende
ax.legend()

# Affichage de la figure
plt.show()

