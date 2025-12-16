import json
import os
import shutil
import random
from urllib.parse import urlparse, parse_qs

# --- Configuration ---
# The name of your Label Studio JSON export file
LABEL_STUDIO_JSON = 'timer_project-6-at-2025-08-03-14-20-054ec1be.json'

# The directory where your original image files are stored
RAW_DATA_DIR = 'timer_images'

# The root directory for the YOLO dataset structure (e.g., 'data' as you requested)
YOLO_ROOT_DIR = 'data'

# The split ratio for training data (e.g., 0.75 for 75% train, 25% val)
TRAIN_SPLIT_RATIO = 0.75

# Class names and their corresponding YOLO class IDs
CLASS_MAPPING = {
    
    'Reading': 0,
    'Unit': 1
}
# ---------------------

def setup_directories():
    """Sets up the required YOLOv8 directory structure."""
    base_dir = os.path.join(YOLO_ROOT_DIR, 'dataset')
    
    # Define subdirectories
    dirs = [
        os.path.join(base_dir, 'train', 'images'),
        os.path.join(base_dir, 'train', 'labels'),
        os.path.join(base_dir, 'val', 'images'),
        os.path.join(base_dir, 'val', 'labels')
    ]
    
    # Create all directories
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print(f"Directory structure created under '{YOLO_ROOT_DIR}/'")
    return base_dir

def get_filename_for_task(task):
    """
    Tries to extract the clean image filename from the Label Studio task data.
    NOTE: You may need to adjust this function if your Label Studio export 
    uses a different path convention (e.g., if 'image' is not present).
    """
    # Assuming 'data' key exists and holds the image path/URL
    if 'data' in task and 'image' in task['data']:
        image_path = task['data']['image']
        
        # 1. Handle typical Label Studio local file path format: 
        #    /data/local-files/?d=some_filename.jpg
        if '?d=' in image_path:
            parsed_url = urlparse(image_path)
            query_params = parse_qs(parsed_url.query)
            if 'd' in query_params:
                return query_params['d'][0]
        
        # 2. Handle cases where it's just a raw filename or simple path
        return os.path.basename(image_path)
    
    # Fallback: If no filename found in 'data', skip this task
    return None

def convert_annotations(tasks, base_dir):
    """Converts Label Studio annotations to YOLO format and handles file split/copy."""
    
    # Shuffle and split tasks
    random.shuffle(tasks)
    train_count = int(len(tasks) * TRAIN_SPLIT_RATIO)
    train_tasks = tasks[:train_count]
    val_tasks = tasks[train_count:]
    
    print(f"\nTotal Tasks found: {len(tasks)}")
    print(f"Splitting: {len(train_tasks)} for Training, {len(val_tasks)} for Validation.")
    
    # Process both train and validation splits
    for split_name, split_tasks in [('train', train_tasks), ('val', val_tasks)]:
        print(f"\nProcessing {split_name} set...")
        
        for task in split_tasks:
            # Task ID for logging/error reporting
            task_id = task.get('id', 'UnknownID')
            
            # 1. Get Filename
            image_filename = get_filename_for_task(task)
            if not image_filename:
                print(f"  [SKIP] Task {task_id}: Could not determine image filename. Skipping.")
                continue

            image_name_only, _ = os.path.splitext(image_filename)
            
            # Paths
            source_image_path = os.path.join(RAW_DATA_DIR, image_filename)
            target_image_path = os.path.join(base_dir, split_name, 'images', image_filename)
            target_label_path = os.path.join(base_dir, split_name, 'labels', f'{image_name_only}.txt')

            # 2. Check for image file existence
            if not os.path.exists(source_image_path):
                print(f"  [ERROR] Task {task_id}: Image not found at '{source_image_path}'. Skipping annotation.")
                continue
                
            # 3. Process Annotations
            yolo_lines = []
            
            # Label Studio stores annotations in task['annotations'][0]['result']
            annotations = task.get('annotations', [{}])[0].get('result', [])
            
            # Annotation result contains pairs of 'label' (rectanglelabels) and 'transcription' (textarea)
            # We filter for 'rectanglelabels' only, as 'textarea' usually holds the text value (R-RES, 08.09)
            
            # Since the user's provided snippet has separate entries for 'label' and 'transcription'
            # with the same ID, we need to iterate over the 'label' entries only.
            
            for item in annotations:
                if item.get('type') == 'rectanglelabels':
                    try:
                        # Extract Bounding Box and Label
                        bbox_value = item['value']
                        label = bbox_value['rectanglelabels'][0]
                        class_id = CLASS_MAPPING.get(label)

                        if class_id is None:
                            print(f"  [WARNING] Task {task_id}: Unknown label '{label}'. Skipping.")
                            continue

                        # Coordinates are typically normalized to 0-100% in Label Studio
                        # We convert them to YOLO format: x_center, y_center, width, height (0.0 to 1.0)
                        x = bbox_value['x'] / 100.0
                        y = bbox_value['y'] / 100.0
                        w = bbox_value['width'] / 100.0
                        h = bbox_value['height'] / 100.0

                        # Format for YOLO: CLASS_ID X_CENTER Y_CENTER WIDTH HEIGHT
                        # Ensure output has sufficient precision for floats
                        yolo_line = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                        yolo_lines.append(yolo_line)

                    except Exception as e:
                        print(f"  [ERROR] Task {task_id} annotation processing failed: {e}. Skipping this annotation.")
                        
            # 4. Write Label File and Copy Image
            if yolo_lines:
                # Write YOLO label file
                with open(target_label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                # Copy image file
                shutil.copy2(source_image_path, target_image_path)
                print(f"  [SUCCESS] Task {task_id}: Copied image and created label '{image_name_only}.txt'")
            else:
                print(f"  [WARNING] Task {task_id}: No valid annotations found. Skipping image copy.")

def create_dataset_yaml(base_dir):
    """Creates the dataset.yaml file for YOLOv8 configuration."""
    yaml_content = f"""
# YOLOv8 dataset.yaml for Label Studio export
# Set 'path' to the root directory containing your 'train' and 'val' folders
path: {YOLO_ROOT_DIR}/dataset/
train: train/images
val: val/images

# number of classes
nc: {len(CLASS_MAPPING)}

# class names
names: {list(CLASS_MAPPING.keys())}
"""
    yaml_path = os.path.join(base_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nCreated YOLO config file: '{yaml_path}'")

def main():
    """Main execution flow."""
    try:
        # Load the Label Studio JSON
        with open(LABEL_STUDIO_JSON, 'r') as f:
            tasks = json.load(f)
            
    except FileNotFoundError:
        print(f"Error: Label Studio JSON file '{LABEL_STUDIO_JSON}' not found.")
        print("Please ensure the file name in the script matches your uploaded file.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file '{LABEL_STUDIO_JSON}'. Check file integrity.")
        return

    # 1. Setup Directories
    base_dir = setup_directories()
    
    # Check raw data directory
    if not os.path.isdir(RAW_DATA_DIR):
        print(f"\nWarning: '{RAW_DATA_DIR}' directory not found.")
        print("Please create this folder and place all your original image files inside it.")
        return

    # 2. Convert and Copy
    convert_annotations(tasks, base_dir)

    # 3. Create dataset.yaml
    create_dataset_yaml(base_dir)

    print("\n\n*** Conversion Complete! ***")
    print(f"Your YOLOv8 dataset is ready in the '{YOLO_ROOT_DIR}/dataset' folder.")

if __name__ == '__main__':
    main()
