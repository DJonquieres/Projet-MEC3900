from ultralytics import YOLO
from pathlib import Path
import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name)
if __name__ == "__main__":
    # model = YOLO('Modele projet/best.pt')
    # datapath = Path('Modele projet/dataset/data.yaml')
    # model.train(
    #     data=str(datapath),
    #     imgsz=640,
    #     epochs=120,
    #     batch=16,
    #     patience=20,    # early stopping
    #     mosaic=1.0,
    #     hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    #     degrees=5, translate=0.1, scale=0.2, shear=2,
    # )
    model = YOLO("trained_model/my_model.pt")  # or path to last.pt

    results = model.predict("path/to/test.mp4", show=True, save=True)


