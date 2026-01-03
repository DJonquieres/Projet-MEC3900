import yaml
import os
import subprocess

"""
Fichier : yaml_file_creation.py

Projet : MEC3900 – Détection et modélisation de la trajectoire d’un ballon de volleyball
Auteur : Denis Jonquieres
Date : 2025-11-05

Description :
Ce script génère automatiquement un fichier data.yaml conforme au format attendu
par YOLOv8 pour l’entraînement d’un modèle de détection.

Principe :
- Les noms des classes sont lus depuis un fichier classes.txt.
- Le nombre de classes (nc) est déduit automatiquement.
- Les chemins vers les ensembles d’entraînement et de validation sont définis.
- Le fichier data.yaml est écrit et peut être utilisé directement pour lancer l’entraînement YOLO.

Lien avec le pipeline du projet :
- Préparation du dataset pour l’entraînement du modèle de détection du ballon
- Garantit la cohérence entre labels, structure des dossiers et configuration YOLO
- Étape clé avant l’apprentissage du réseau de neurones
"""


def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    """
    Crée un fichier data.yaml à partir d’un fichier classes.txt.

    Paramètres
    ----------
    path_to_classes_txt : str
        Chemin vers le fichier classes.txt contenant une classe par ligne.
        Exemple :
            ball
    path_to_data_yaml : str
        Chemin de sortie du fichier data.yaml généré.

    Fonctionnement
    --------------
    1) Lecture des noms de classes depuis classes.txt,
    2) Construction du dictionnaire de configuration YOLO,
    3) Écriture du fichier YAML sur disque.

    Notes
    -----
    Le fichier data.yaml est requis par Ultralytics YOLOv8 pour :
    - connaître le nombre de classes (nc),
    - associer les identifiants de classes à des noms lisibles,
    - localiser les dossiers train/validation.
    """

    # Vérification de l’existence du fichier classes.txt
    if not os.path.exists(path_to_classes_txt):
        print(
            f'classes.txt file not found! '
            f'Please create a classes.txt labelmap and move it to {path_to_classes_txt}'
        )
        return

    # Lecture des noms de classes (une classe par ligne non vide)
    classes = []
    with open(path_to_classes_txt, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            classes.append(line.strip())

    number_of_classes = len(classes)

    # Création du dictionnaire de configuration YOLO
    # Les chemins sont relatifs à la clé 'path'
    data = {
        'path': 'data',
        'train': 'train/images',
        'val': 'validation/images',
        'nc': number_of_classes,
        'names': classes
    }

    # Écriture du fichier YAML
    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f'Created config file at {path_to_data_yaml}')
    return


# -------------------------------------------------------------------
# Exécution du script
# -------------------------------------------------------------------

# Chemins vers le fichier des classes et vers le YAML à générer
path_to_classes_txt = 'dataset/classes.txt'
path_to_data_yaml = 'data.yaml'

# Génération du fichier data.yaml
create_data_yaml(path_to_classes_txt, path_to_data_yaml)

# Affichage du contenu du fichier pour validation rapide
print('\nFile contents:\n')

# Commande système pour afficher data.yaml (Windows : "type")
# NOTE : dépend du système d’exploitation
subprocess.run("type data.yaml", shell=True)
