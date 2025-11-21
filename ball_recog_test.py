from ultralytics import YOLO
import numpy as np
import cv2
import csv

if __name__ == "__main__":
    model = YOLO("runs/detect/train2/weights/best.pt")  # or path to last.pt
    BALL_CLASS_ID = 0

    video_path = "path/to/test.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("match_with_trajectory.mp4", fourcc, fps, (width, height))

    data = []
    trajectory_points = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        results = model.predict(frame, verbose=False)[0]
        boxes = results.boxes
        ball_center = None

        if boxes is not None and len(boxes)>0:
            best_conf = 0.0
            best_center = None

            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                if int(cls) != BALL_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.tolist()

                cx = (x1 + x2)/2
                cy = (y1 + y2)/2

                if conf > best_conf:
                    best_conf = float(conf)
                    best_center = (cx,cy)

                t_sec = frame_idx / fps
                
                data.append([frame_idx, t_sec, cx, cy, float(conf)])

            if best_center is not None:
                ball_center = best_center
                trajectory_points.append(ball_center)
            
        if len(trajectory_points) >= 2:
            pts = np.array(trajectory_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame,[pts], isClosed=False, color=(0,0,255), thickness=2)
        
        if ball_center is not None:
            cx, cy = ball_center
            cv2.circle(frame,(int(cx), int(cy)), radius=6, color=(0,0,255), thickness=-1)
        
        out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()
    print("Done! Saved video as match_with_trajectory.mp4")

    with open("ball_positions_csv", "w", newline="") as fwrite:
        writer = csv.writer(fwrite)
        writer.writerow(["frame", "time_s", "x_px", "y_px", "confidence"])
        writer.writerows(data)

    print("Saved ball positions to ball positions to ball_positions.csv")

    

