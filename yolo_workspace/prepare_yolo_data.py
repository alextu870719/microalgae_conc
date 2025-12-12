import os
import json
import shutil
import glob
import random
import numpy as np

def convert_labelme_to_yolo(json_path, output_path, class_map):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_height = data['imageHeight']
    image_width = data['imageWidth']
    
    yolo_lines = []
    
    for shape in data['shapes']:
        label = shape['label']
        if label not in class_map:
            print(f"Warning: Unknown label '{label}' in {json_path}")
            continue
        
        class_id = class_map[label]
        points = shape['points']
        
        # LabelMe stores points as [[x1, y1], [x2, y2]] for rectangles
        # But sometimes it might be polygon, we'll assume rectangle or polygon and get bounding box
        points = np.array(points)
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        
        # Normalize
        center_x = (x_min + x_max) / 2.0 / image_width
        center_y = (y_min + y_max) / 2.0 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        
        yolo_lines.append(f"{class_id} {center_x} {center_y} {width} {height}")
        
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

def prepare_dataset(source_dir, output_dir, split_ratio=0.8):
    # Setup directories
    for split in ['train', 'val']:
        for dtype in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, dtype), exist_ok=True)
            
    # Get all json files
    json_files = glob.glob(os.path.join(source_dir, "*.json"))
    print(f"Found {len(json_files)} labeled images.")
    
    # Shuffle
    random.shuffle(json_files)
    
    # Split
    split_idx = int(len(json_files) * split_ratio)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]
    
    class_map = {'cell': 0, 'algae': 0, 'tiny green ball': 0} # Support multiple names
    
    for files, split in [(train_files, 'train'), (val_files, 'val')]:
        for json_file in files:
            # Image file path (assume jpg)
            img_file = json_file.replace('.json', '.jpg')
            if not os.path.exists(img_file):
                print(f"Warning: Image file {img_file} not found for {json_file}")
                continue
                
            # Copy image
            filename = os.path.basename(img_file)
            shutil.copy(img_file, os.path.join(output_dir, split, 'images', filename))
            
            # Convert and save label
            label_filename = os.path.splitext(filename)[0] + '.txt'
            convert_labelme_to_yolo(
                json_file, 
                os.path.join(output_dir, split, 'labels', label_filename),
                class_map
            )
            
    print(f"Dataset prepared at {output_dir}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create data.yaml
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images

nc: 1
names: ['cell']
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print("Created data.yaml")

if __name__ == "__main__":
    source = "yolo_workspace/images_to_label"
    output = "yolo_workspace/dataset"
    prepare_dataset(source, output)
