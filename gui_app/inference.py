import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class MicroalgaeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.crop_size = 640
        self.overlap = 0.2
        self.conf_thres = 0.05

    def predict(self, image_path, roi_points=None, conf_thres=0.05):
        """
        image_path: Path to image
        roi_points: List of (x, y) tuples defining the polygon. If None, use full image.
        conf_thres: Confidence threshold for this prediction
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w, _ = img.shape
        
        # Define the area to scan
        if roi_points and len(roi_points) > 2:
            poly = Polygon(roi_points)
            min_x, min_y, max_x, max_y = poly.bounds
            min_x, min_y = max(0, int(min_x)), max(0, int(min_y))
            max_x, max_y = min(w, int(max_x)), min(h, int(max_y))
            scan_area = (min_x, min_y, max_x, max_y)
        else:
            poly = None
            scan_area = (0, 0, w, h)

        # Sliding window
        step = int(self.crop_size * (1 - self.overlap))
        all_boxes = []
        
        start_x, start_y, end_x, end_y = scan_area

        for y in range(start_y, end_y, step):
            for x in range(start_x, end_x, step):
                # Adjust crop coordinates
                x1 = x
                y1 = y
                x2 = min(x + self.crop_size, w)
                y2 = min(y + self.crop_size, h)
                
                # If crop is smaller than crop_size and we can shift back, do it
                if x2 - x1 < self.crop_size and x1 > 0:
                    x1 = max(0, x2 - self.crop_size)
                if y2 - y1 < self.crop_size and y1 > 0:
                    y1 = max(0, y2 - self.crop_size)
                
                crop = img[y1:y2, x1:x2]
                
                # Inference
                results = self.model(crop, verbose=False, conf=conf_thres)
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # Global coordinates
                        gx1 = bx1 + x1
                        gy1 = by1 + y1
                        gx2 = bx2 + x1
                        gy2 = by2 + y1
                        
                        # Filter by ROI if exists
                        if poly:
                            center_x = (gx1 + gx2) / 2
                            center_y = (gy1 + gy2) / 2
                            if not poly.contains(Point(center_x, center_y)):
                                continue
                        
                        all_boxes.append([gx1, gy1, gx2, gy2, conf])

        # NMS
        if not all_boxes:
            return [], img

        boxes_array = np.array(all_boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        scores = boxes_array[:, 4]
        
        boxes_wh = []
        for i in range(len(boxes_array)):
            boxes_wh.append([x1[i], y1[i], x2[i]-x1[i], y2[i]-y1[i]])
            
        # Convert scores to list to satisfy some opencv versions/linters
        scores_list = scores.tolist()
        
        indices = cv2.dnn.NMSBoxes(boxes_wh, scores_list, score_threshold=conf_thres, nms_threshold=0.4)
        
        final_detections = []
        for i in indices:
            idx = i
            box = boxes_array[idx]
            final_detections.append(box) # [x1, y1, x2, y2, conf]
            
        return final_detections, img

    def draw_results(self, img, detections, draw_labels=True):
        output_img = img.copy()
        if not draw_labels:
            return output_img
            
        for i, box in enumerate(detections):
            x1, y1, x2, y2, conf = box
            # Draw rectangle
            cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label number
            label = str(i + 1)
            font_scale = 0.6
            thickness = 2
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw background for text for better visibility
            cv2.rectangle(output_img, (int(x1), int(y1) - 20), (int(x1) + w, int(y1)), (0, 255, 0), -1)
            cv2.putText(output_img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
        return output_img
