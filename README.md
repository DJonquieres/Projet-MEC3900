# Projet-MEC3900
Detection et modélisation de la trajectoire d'un ballon de volleyball à partir d'une vidéo

Les 3 modèles de projet (v1, v2, v3) sont entrainés avec différents datasets

Les dossiers sont composés de sous-dossier qui indiquent les éléments respectifs de chaque modèle: tout les programmes python sont dans le sous-dossier "code", les résultat de sortie des codes dans le sous-dossier "resultats", et les images ainsi que labels utilisés dans le sous-dossier "dataset". Certains éléments n'ont pas été inclus dans le commit (comme les vidéos ou une partie des données d'entrainement) pour éviter d'allourdir le commit.

Modèle_projet_v1:
- Premier modèle entrainé avec un dataset trouvé sur kaggle (peu de variation des images)
- Dataset: https://www.kaggle.com/datasets/pythonistasamurai/volleyball-ball-object-detection-dataset


Modèle_projet_v2:
- Modèle principal utilisé pour les démonstrations du rapport final et de la présentation oral
- Dataset: créer manuellement à partir de screenshots de vidéos youtube (225 images)

Modèle_projet_v3:
- Dernier modèle entrainer avec un dataset trouvé sur roboflow (beaucoup de variation des images)
- Dataset: https://universe.roboflow.com/project-9c0mz/volleyballdetection-t9alv

Les deux dossier restant, Modèle_coco et Demo_3D sont des dossiers ou j'ai jouer avec différents éléments du système (modèle YOLO et conversion en 3D respectivement) et ne sont donc pas annotés.

Il est finalement important de noter que le code à été réorganisé et beaucoup annoté depuis sa complétion donc il est possible que certains path dans les codes ait changés et ne fonctionnent plus. Il suffit simplement de modifier le path si ceci est le cas


** Note: L'annotation des programmes a été en parti faite avec ChatGPT **


