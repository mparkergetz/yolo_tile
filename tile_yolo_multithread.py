import os
import cv2
import numpy as np
import concurrent.futures  # Multithreading

def tile_image(image_path, label_path, output_dir, tile_size=640, overlap=0.25):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return

    img_h, img_w, _ = img.shape
    stride = int(tile_size * (1 - overlap))  # Calculate stride
    tile_id = 0

    # Read YOLO annotation file
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                data = line.strip().split()
                class_id = int(data[0])
                x_center, y_center, width, height = map(float, data[1:])
                bboxes.append((class_id, x_center * img_w, y_center * img_h, width * img_w, height * img_h))

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for y in range(0, img_h, stride):
        if y + tile_size > img_h:
            y = img_h - tile_size  # Adjust last row tile to align with the bottom edge

        for x in range(0, img_w, stride):
            if x + tile_size > img_w:
                x = img_w - tile_size  # Adjust last column tile to align with the right edge

            # Crop tile with adjusted x and y
            tile = img[y:y + tile_size, x:x + tile_size]

            tile_filename = f"{base_name}_{tile_id}.jpg"
            tile_path = os.path.join(output_dir, tile_filename)
            cv2.imwrite(tile_path, tile)

            # Process annotations for this tile
            tile_bboxes = []
            for class_id, xc, yc, w, h in bboxes:
                x_min, y_min = xc - w / 2, yc - h / 2
                x_max, y_max = xc + w / 2, yc + h / 2

                # Check if bbox is within the tile
                if x_max > x and x_min < x + tile_size and y_max > y and y_min < y + tile_size:
                    # Clip bbox to tile boundaries
                    x_min_clipped = max(x_min, x)
                    y_min_clipped = max(y_min, y)
                    x_max_clipped = min(x_max, x + tile_size)
                    y_max_clipped = min(y_max, y + tile_size)

                    new_w = x_max_clipped - x_min_clipped
                    new_h = y_max_clipped - y_min_clipped
                    new_xc = (x_min_clipped + x_max_clipped) / 2 - x
                    new_yc = (y_min_clipped + y_max_clipped) / 2 - y

                    # Filtering Criteria
                    original_area = w * h
                    sliced_area = new_w * new_h
                    MIN_ABS_SIZE = 16  # Minimum size for small objects
                    aspect_ratio = max(new_w, new_h) / max(1, min(new_w, new_h))  # Avoid division by zero

                    # Compute IoU with original bbox
                    xA = max(x_min, x_min_clipped)
                    yA = max(y_min, y_min_clipped)
                    xB = min(x_max, x_max_clipped)
                    yB = min(y_max, y_max_clipped)
                    intersect_area = max(0, xB - xA) * max(0, yB - yA)
                    iou = intersect_area / (original_area + 1e-6)  # Avoid division by zero

                    # Apply Filtering:
                    if (sliced_area >= 0.1 * original_area and  # Keep at least 10% of the original bbox
                        new_w >= MIN_ABS_SIZE and new_h >= MIN_ABS_SIZE and  # Avoid tiny boxes
                        aspect_ratio <= 4.0 and  # Ensure reasonable shape
                        iou >= 0.2):  # Ensure meaningful overlap with original bbox
                        # Normalize bbox for YOLO format
                        tile_bboxes.append((class_id, new_xc / tile_size, new_yc / tile_size, new_w / tile_size, new_h / tile_size))

            # Save annotations
            tile_label_path = tile_path.replace(".jpg", ".txt")
            with open(tile_label_path, "w") as f:
                for bbox in tile_bboxes:
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")

            tile_id += 1

def process_image(file, input_dir, output_dir):
    """ Wrapper function for multithreading image processing """
    if file.endswith(".jpg"):
        img_path = os.path.join(input_dir, file)
        label_path = os.path.join(input_dir, file.replace(".jpg", ".txt"))
        tile_image(img_path, label_path, output_dir)

# Paths
input_dir = "val"
output_dir = "val_sliced"
os.makedirs(output_dir, exist_ok=True)

# Get list of images
image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

# Use multithreading to process images in parallel
num_workers = min(20, os.cpu_count())  
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(lambda file: process_image(file, input_dir, output_dir), image_files)

