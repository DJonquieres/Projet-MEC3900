from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/train2/weights/best.pt")

    metrics = model.val()   
    print("\n===== METRICS =====")
    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.5:0.95:", metrics.box.map)   
