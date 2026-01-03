import pathlib
import torch
from torch.serialization import add_safe_globals

# 1) Hack: faire en sorte que WindowsPath fonctionne sur Mac
class _FakeWindowsPath(pathlib.PosixPath):
    pass

pathlib.WindowsPath = _FakeWindowsPath

# 2) Importer la classe de modèle YOLO et l'autoriser pour le unpickling
from ultralytics.nn.tasks import DetectionModel

add_safe_globals([DetectionModel])

# 3) Charger le checkpoint EN DÉSACTIVANT weights_only
#    (ok ici car le fichier vient de toi)
ckpt = torch.load("../yolo_models/best.pt", map_location="cpu", weights_only=False)

# 4) Nettoyage minimal : enlever les train_args qui contiennent souvent des Paths
if "train_args" in ckpt:
    ckpt["train_args"] = None

# 5) Sauvegarder un checkpoint "propre" pour Mac
torch.save(ckpt, "best_mac.pt")

print("Checkpoint nettoyé et sauvegardé sous best_mac.pt")
