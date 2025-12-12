import sys
import os
import cv2
import numpy as np
import pandas as pd

# Add the cloned repository to the python path
sys.path.append(os.path.abspath("microalgae_conc"))

from CellCounter.Segmentator import detect_cells, visualize_circles

def run_segmentation(image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.basename(image_path)
    print(f"Processing {filename}...")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}")
        return

    # Using parameters similar to inference_example.py but adjusted based on our previous experiments
    # inference_example: dist=10, min_radius=3, max_radius=20, sensitivity=20, blur=3
    cells = detect_cells(
        img,
        minDist=12,       # Similar to our best result
        minRadius=4,      # Similar to our best result
        maxRadius=40,     # Similar to our best result
        param2=23,        # Similar to our best result
        blur_kernel=3     # Similar to our best result
    )

    if cells is not None:
        count = cells.shape[1]
        print(f"Detected {count} cells in {filename}")
        
        # Convert BGR to RGB for visualization (matplotlib uses RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        output_path = os.path.join(output_dir, f"result_{filename}")
        visualize_circles(img_rgb, cells, output_path)
        print(f"Saved result to {output_path}")
    else:
        print(f"No cells detected in {filename}")

if __name__ == "__main__":
    # Process all jpeg files in the current directory
    files = [f for f in os.listdir('.') if f.endswith('.jpeg') or f.endswith('.jpg')]
    output_dir = "github_results"
    
    for f in files:
        if "result" not in f: # Avoid processing result images
            run_segmentation(f, output_dir)
