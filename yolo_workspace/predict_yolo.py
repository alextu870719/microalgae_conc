from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob

def predict_large_image(model_path, image_path, output_path, crop_size=640, overlap=0.2):
    # Load model
    model = YOLO(model_path)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}")
        return

    h, w, _ = img.shape
    print(f"Processing {os.path.basename(image_path)} ({w}x{h})...")
    
    # Step size with overlap
    step = int(crop_size * (1 - overlap))
    
    all_boxes = []
    
    # Sliding window
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Calculate crop coordinates
            x1 = x
            y1 = y
            x2 = min(x + crop_size, w)
            y2 = min(y + crop_size, h)
            
            # If we are at the edge, adjust x1/y1 to keep crop_size if possible
            if x2 - x1 < crop_size and x1 > 0:
                x1 = max(0, x2 - crop_size)
            if y2 - y1 < crop_size and y1 > 0:
                y1 = max(0, y2 - crop_size)
                
            crop = img[y1:y2, x1:x2]
            
            # Run inference
            results = model(crop, verbose=False, conf=0.05) # Adjust conf as needed
            
            # Map boxes back to original image coordinates
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Box coordinates in crop
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Map to global
                    gx1 = bx1 + x1
                    gy1 = by1 + y1
                    gx2 = bx2 + x1
                    gy2 = by2 + y1
                    
                    all_boxes.append([gx1, gy1, gx2, gy2, conf])

    # Apply NMS (Non-Maximum Suppression) to merge overlapping boxes from different crops
    if not all_boxes:
        print("No cells detected.")
        return

    boxes_array = np.array(all_boxes)
    
    # OpenCV NMS
    # boxes: (x, y, w, h) for NMSBoxes
    # But we have x1, y1, x2, y2
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    scores = boxes_array[:, 4]
    
    boxes_wh = []
    for i in range(len(boxes_array)):
        boxes_wh.append([x1[i], y1[i], x2[i]-x1[i], y2[i]-y1[i]])
        
    indices = cv2.dnn.NMSBoxes(boxes_wh, scores, score_threshold=0.05, nms_threshold=0.4)
    
    final_count = len(indices)
    print(f"Detected {final_count} cells.")
    
    # Draw results
    output_img = img.copy()
    for i in indices:
        idx = i # NMSBoxes returns a list of indices, sometimes wrapped
        box = boxes_array[idx]
        bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        cv2.rectangle(output_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        # cv2.circle(output_img, (int((bx1+bx2)/2), int((by1+by2)/2)), 2, (0, 0, 255), -1)

    cv2.imwrite(output_path, output_img)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    # Path to the best trained model
    # Note: The run name might change (microalgae_yolov8, microalgae_yolov82...), I'll try to find the latest
    runs_dir = "yolo_workspace/runs"
    # Find the latest run directory
    subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    latest_run = max(subdirs, key=os.path.getmtime)
    model_path = os.path.join(latest_run, "weights", "best.pt")
    
    print(f"Using model: {model_path}")
    
    output_dir = "yolo_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Process all jpeg files
    files = [f for f in os.listdir('.') if (f.endswith('.jpeg') or f.endswith('.jpg')) and "result" not in f]
    
    for f in files:
        predict_large_image(model_path, f, os.path.join(output_dir, f"yolo_{f}"))
