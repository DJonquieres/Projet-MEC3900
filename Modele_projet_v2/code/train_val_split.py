from pathlib import Path
import random
import os
import sys
import shutil
import argparse

"""
Fichier : train_val_split.py

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-11-05

Description :
Ce script permet de séparer automatiquement un dataset annoté au format YOLO
(images + fichiers .txt) en deux sous-ensembles :
- un ensemble d’entraînement (train),
- un ensemble de validation (validation).

La séparation est effectuée de manière aléatoire selon un ratio défini par l’utilisateur.
La structure de sortie est compatible avec l’entraînement d’un modèle YOLOv8.

Lien avec le projet :
- Étape de préparation des données (dataset)
- Condition nécessaire à l’entraînement fiable du modèle de détection du ballon
- Contribue à la robustesse et à la reproductibilité de l’apprentissage

Structure attendue en entrée :
datapath/
 ├─ images/
 │   ├─ image1.jpg
 │   ├─ image2.jpg
 │   └─ ...
 └─ labels/
     ├─ image1.txt
     ├─ image2.txt
     └─ ...

Structure générée en sortie :
data/
 ├─ train/
 │   ├─ images/
 │   └─ labels/
 └─ validation/
     ├─ images/
     └─ labels/

Remarque :
Les images sans fichier .txt associé sont considérées comme des images de fond
(background) et sont copiées sans annotation.
"""

# -------------------------------------------------------------------
# Définition et lecture des arguments utilisateur (ligne de commande)
# -------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    '--datapath',
    help='Path to data folder containing image and annotation files',
    required=True
)
parser.add_argument(
    '--train_pct',
    help='Ratio of images to go to train folder; the rest go to validation folder (example: ".8")',
    default=.8
)

args = parser.parse_args()

data_path = args.datapath
train_percent = float(args.train_pct)

# -------------------------------------------------------------------
# Vérification des paramètres d’entrée
# -------------------------------------------------------------------
# Ces vérifications évitent de lancer un split invalide ou incohérent.

if not os.path.isdir(data_path):
   print(
       'Directory specified by --datapath not found. '
       'Verify the path is correct (and uses double back slashes if on Windows) and try again.'
   )
   sys.exit(0)

if train_percent < .01 or train_percent > 0.99:
   print('Invalid entry for train_pct. Please enter a number between .01 and .99.')
   sys.exit(0)

# Pour information : proportion du dataset réservée à la validation
val_percent = 1 - train_percent

# -------------------------------------------------------------------
# Définition des chemins du dataset brut (entrée)
# -------------------------------------------------------------------

input_image_path = os.path.join(data_path, 'images')
input_label_path = os.path.join(data_path, 'labels')

# -------------------------------------------------------------------
# Définition des chemins de sortie (structure YOLO standard)
# -------------------------------------------------------------------

cwd = os.getcwd()
train_img_path = os.path.join(cwd, 'data/train/images')
train_txt_path = os.path.join(cwd, 'data/train/labels')
val_img_path = os.path.join(cwd, 'data/validation/images')
val_txt_path = os.path.join(cwd, 'data/validation/labels')

# -------------------------------------------------------------------
# Création des dossiers de sortie s’ils n’existent pas
# -------------------------------------------------------------------
# Cette étape rend le script robuste à une première exécution.

for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
   if not os.path.exists(dir_path):
      os.makedirs(dir_path)
      print(f'Created folder at {dir_path}.')

# -------------------------------------------------------------------
# Lecture de la liste des fichiers images et annotations disponibles
# -------------------------------------------------------------------

# rglob('*') permet d’inclure tous les formats d’images présents dans le dossier
img_file_list = [path for path in Path(input_image_path).rglob('*')]
txt_file_list = [path for path in Path(input_label_path).rglob('*')]

print(f'Number of image files: {len(img_file_list)}')
print(f'Number of annotation files: {len(txt_file_list)}')

# -------------------------------------------------------------------
# Calcul du nombre d’images par sous-ensemble
# -------------------------------------------------------------------
# La séparation se fait uniquement sur les images.
# Les labels sont copiés uniquement s’ils existent.

file_num = len(img_file_list)
train_num = int(file_num * train_percent)
val_num = file_num - train_num

print('Images moving to train: %d' % train_num)
print('Images moving to validation: %d' % val_num)

# -------------------------------------------------------------------
# Sélection aléatoire et copie des fichiers
# -------------------------------------------------------------------
# Principe :
# - On choisit aléatoirement une image,
# - On copie l’image et son label associé (si présent),
# - On retire l’image de la liste pour éviter les doublons.

for i, set_num in enumerate([train_num, val_num]):
  for ii in range(set_num):

    # Sélection aléatoire d’une image restante
    img_path = random.choice(img_file_list)
    img_fn = img_path.name
    base_fn = img_path.stem

    # Le label YOLO doit avoir le même nom que l’image
    txt_fn = base_fn + '.txt'
    txt_path = os.path.join(input_label_path, txt_fn)

    # Choix du dossier cible selon l’itération :
    # i == 0 -> train
    # i == 1 -> validation
    if i == 0:
      new_img_path, new_txt_path = train_img_path, train_txt_path
    elif i == 1:
      new_img_path, new_txt_path = val_img_path, val_txt_path

    # Copie de l’image vers le dossier cible
    shutil.copy(img_path, os.path.join(new_img_path, img_fn))
    # (On utilise copy plutôt que move pour conserver le dataset original intact)

    # Copie du fichier d’annotation seulement s’il existe
    # Si le fichier n’existe pas, l’image est considérée comme "background"
    if os.path.exists(txt_path):
      shutil.copy(txt_path, os.path.join(new_txt_path, txt_fn))

    # Suppression de l’image de la liste source
    # Garantit qu’une image ne sera pas copiée deux fois
    img_file_list.remove(img_path)

