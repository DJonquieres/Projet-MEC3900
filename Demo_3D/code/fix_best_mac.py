import pathlib
import torch
from torch.serialization import add_safe_globals

# 1) Hack WindowsPath -> PosixPath pour Mac
class _FakeWindowsPath(pathlib.PosixPath):
    pass

pathlib.WindowsPath = _FakeWindowsPath

# 2) Autoriser la classe DetectionModel pour le unpickling
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])

# 3) Charger le checkpoint déjà 'best_mac.pt'
ckpt = torch.load("../yolo_models/best_mac.pt", map_location="cpu", weights_only=False)

# 4) Si train_args est None, on le remplace par un dict vide
if ckpt.get("train_args") is None:
    ckpt["train_args"] = {}

# 5) Sauvegarder sous un nouveau nom
torch.save(ckpt, "best_mac2.pt")

print("Checkpoint corrigé et sauvegardé sous best_mac2.pt")
