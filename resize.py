# =================================================================
# PROFESSIONAL BATCH IMAGE RESIZING SCRIPT
# This script resizes all images in a folder to a target width
# while maintaining the original aspect ratio.
# =================================================================
import cv2
import os
import glob

# --- CONFIGURATION (YOU MUST EDIT THESE VALUES) ---

# Set the full path to your source folder and the destination for resized images.
# IMPORTANT: Use forward slashes '/' or double backslashes '\\' in paths.
INPUT_FOLDER = 'C:/Users/User/Downloads/FactoryNext/New folder'
OUTPUT_FOLDER = 'C:/Users/User/Downloads/FactoryNext/IR_Resized_for_Labeling'

# Define the target width for your images in pixels.
# The height will be calculated automatically to keep the aspect ratio.
# A value between 800 and 1200 is usually good for labeling.
TARGET_WIDTH = 1024

# --- SCRIPT START ---
# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Find all common image types in the input folder
image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.jpg')) + \
              glob.glob(os.path.join(INPUT_FOLDER, '*.png')) + \
              glob.glob(os.path.join(INPUT_FOLDER, '*.jpeg'))

if not image_paths:
    print(f"ERROR: No images found in '{INPUT_FOLDER}'. Please check the path.")
    exit()

total_images = len(image_paths)
print(f"Found {total_images} images to resize to a width of {TARGET_WIDTH} pixels.")

# Loop through every image found
for i, image_path in enumerate(image_paths):
    original_filename = os.path.basename(image_path)
    
    # Read the original image from the disk
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"  [{i+1}/{total_images}] WARNING: Could not read {original_filename}. Skipping.")
        continue
    
    print(f"--- [{i+1}/{total_images}] Processing {original_filename} ---")
    
    # --- Resizing Logic ---
    # Get the original image dimensions
    (original_height, original_width) = img.shape[:2]
    
    # Calculate the aspect ratio
    aspect_ratio = original_height / original_width
    
    # Calculate the new height based on the target width and aspect ratio
    target_height = int(TARGET_WIDTH * aspect_ratio)
    
    # Define the new dimensions
    new_dimensions = (TARGET_WIDTH, target_height)
    
    # Resize the image using a high-quality interpolation method
    resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
    
    # Define the output path
    output_path = os.path.join(OUTPUT_FOLDER, original_filename)
    
    # Save the resized image to the output folder
    cv2.imwrite(output_path, resized_img)
    print(f"  > Resized to {TARGET_WIDTH}x{target_height} and saved.")

print(f"\n\n--- BATCH RESIZING COMPLETE ---")
print(f"Successfully resized {total_images} images and saved them in '{OUTPUT_FOLDER}'.")