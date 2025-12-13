import cv2
import numpy as np
import onnxruntime as ort
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class MicroalgaeDetector:
    def __init__(self, model_path):
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get input details
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # Get output details
        model_outputs = self.session.get_outputs()
        self.output_name = model_outputs[0].name
        
        self.crop_size = 640 # Sliding window size (should match model input usually, or be handled)
        self.overlap = 0.2

    def preprocess(self, img):
        """
        Preprocess image for YOLOv8 ONNX model.
        Returns: preprocessed_img, ratio, (dw, dh)
        """
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = (self.input_width, self.input_height)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        img = img.astype(np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img[None]  # expand for batch dim
        
        return img, ratio, (dw, dh)

    def predict(self, image_path, roi_points=None, conf_thres=0.05):
        """
        image_path: Path to image
        roi_points: List of (x, y) tuples defining the polygon. If None, use full image.
        conf_thres: Confidence threshold for this prediction
        """
        # Read image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
        else:
            img = image_path

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
        # Note: If the image is very large, we slide. If small, we just resize.
        # For consistency with previous logic, we keep sliding window if implemented, 
        # but standard YOLO usually resizes the whole image. 
        # The previous implementation used sliding window with crop_size=640.
        # We will maintain that logic.
        
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
                
                # Preprocess
                blob, ratio, (dw, dh) = self.preprocess(crop)
                
                # Inference
                outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
                
                # Postprocess
                # Output shape [1, 5, 8400] -> [1, 4+1, 8400]
                # Transpose to [1, 8400, 5]
                predictions = np.transpose(outputs, (0, 2, 1))
                prediction = predictions[0] # [8400, 5]
                
                # Filter by confidence
                # Format: x, y, w, h, conf
                scores = prediction[:, 4]
                mask = scores > conf_thres
                filtered_preds = prediction[mask]
                
                if len(filtered_preds) == 0:
                    continue
                    
                # Convert boxes to original crop coordinates
                # Current boxes are in resized 640x640 coordinates (with padding)
                # We need to map them back to the crop size
                
                boxes = filtered_preds[:, :4]
                confs = filtered_preds[:, 4]
                
                # xywh to xyxy
                # x, y are center coordinates
                bx = boxes[:, 0]
                by = boxes[:, 1]
                bw = boxes[:, 2]
                bh = boxes[:, 3]
                
                x1_box = bx - bw / 2
                y1_box = by - bh / 2
                x2_box = bx + bw / 2
                y2_box = by + bh / 2
                
                # Remove padding
                x1_box -= dw
                y1_box -= dh
                x2_box -= dw
                y2_box -= dh
                
                # Scale back
                x1_box /= ratio[0]
                y1_box /= ratio[1]
                x2_box /= ratio[0]
                y2_box /= ratio[1]
                
                # Clip to crop dimensions
                crop_h, crop_w = crop.shape[:2]
                x1_box = np.clip(x1_box, 0, crop_w)
                y1_box = np.clip(y1_box, 0, crop_h)
                x2_box = np.clip(x2_box, 0, crop_w)
                y2_box = np.clip(y2_box, 0, crop_h)
                
                for i in range(len(confs)):
                    # Global coordinates
                    gx1 = x1_box[i] + x1
                    gy1 = y1_box[i] + y1
                    gx2 = x2_box[i] + x1
                    gy2 = y2_box[i] + y1
                    conf = confs[i]
                    
                    # Filter by ROI if exists
                    if poly:
                        center_x = (gx1 + gx2) / 2
                        center_y = (gy1 + gy2) / 2
                        if not poly.contains(Point(center_x, center_y)):
                            continue
                    
                    all_boxes.append([gx1, gy1, gx2, gy2, conf])

        # Global NMS
        if not all_boxes:
            return [], img

        boxes_array = np.array(all_boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        scores = boxes_array[:, 4]
        
        # NMS expects xywh
        boxes_wh = []
        for i in range(len(boxes_array)):
            boxes_wh.append([x1[i], y1[i], x2[i]-x1[i], y2[i]-y1[i]])
            
        scores_list = scores.tolist()
        
        indices = cv2.dnn.NMSBoxes(boxes_wh, scores_list, score_threshold=conf_thres, nms_threshold=0.4)
        
        final_detections = []
        if len(indices) > 0:
            # Flatten indices if needed
            indices = indices.flatten()
            for i in indices:
                box = boxes_array[i]
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
