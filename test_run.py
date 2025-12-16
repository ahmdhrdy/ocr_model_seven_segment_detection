import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow.lite as tflite
import string

# --- CONFIGURATION ---
DETECTOR_MODEL_PATH = 'best_float32.tflite'
OCR_MODEL_PATH = '7seg_model_float16.tflite'
INPUT_IMAGES_FOLDER = 'test_images'
OUTPUT_RESULTS_FOLDER = 'results'
FAILED_FOLDER = 'failed_detections'

os.makedirs(OUTPUT_RESULTS_FOLDER, exist_ok=True)
os.makedirs(FAILED_FOLDER, exist_ok=True)

# --- HELPER FUNCTIONS ---
def prepare_input(image_array):
  input_data = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
  input_data = cv2.resize(input_data, (200, 31))
  input_data = input_data[np.newaxis]
  input_data = np.expand_dims(input_data, 3)
  input_data = input_data.astype('float32')/255
  return input_data

def predict_ocr(prepared_data, model_path):
  interpreter = tflite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  interpreter.set_tensor(input_details[0]['index'], prepared_data)
  interpreter.invoke()
  output = interpreter.get_tensor(output_details[0]['index'])
  return output

# --- MAIN PROCESSING ---
print("Starting batch processing...")

print(f"Loading detector model: {DETECTOR_MODEL_PATH}")
detector_model = YOLO(DETECTOR_MODEL_PATH)

image_files = [f for f in os.listdir(INPUT_IMAGES_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
total_images = len(image_files)
print(f"Found {total_images} images to process.")

for i, image_file in enumerate(image_files):
    print(f"\n--- [{i+1}/{total_images}] Processing: {image_file} ---")
    image_path = os.path.join(INPUT_IMAGES_FOLDER, image_file)

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"  ERROR: Could not read image file: {image_file}. Skipping.")
        continue

    results = detector_model.predict(source=original_image, imgsz=512, verbose=False)
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        xyxy = boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        cropped_screen = original_image[y1:y2, x1:x2]
        print(f"  Stage 1 Success: Screen detected.")

        alphabet = string.digits + '.'
        blank_index = len(alphabet)
        prepared_data = prepare_input(cropped_screen)
        result = predict_ocr(prepared_data, OCR_MODEL_PATH)
        
        raw_text = "".join(alphabet[index] for index in result[0] if index not in [blank_index, -1])
        print(f"  Stage 2 Success: Extracted text is '{raw_text}'")

        # --- THIS IS THE NEW, FILENAME LOGIC ---
        # Sanitize the predicted text to make it a valid base filename
        filename_prediction = raw_text.replace('.', '_')
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        filename_base = "".join(c for c in filename_prediction if c in valid_chars)
        
        if not filename_base:
            filename_base = "prediction_failed"
        
        # Check if a file with this name already exists and add a counter if it does
        output_path = os.path.join(OUTPUT_RESULTS_FOLDER, f"{filename_base}.png")
        counter = 1
        while os.path.exists(output_path):
            # If '2840.png' exists, the next name will be '2840_(1).png', then '2840_(2).png', etc.
            output_path = os.path.join(OUTPUT_RESULTS_FOLDER, f"{filename_base}_({counter}).png")
            counter += 1
        # --- END OF NEW LOGIC ---

        cv2.imwrite(output_path, cropped_screen)
        print(f"  Result saved to: {output_path}")

    else:
        print(f"  Stage 1 FAILED: No screen detected in {image_file}.")
        shutil.copy(image_path, FAILED_FOLDER)
        print(f"  Original image has been copied to '{FAILED_FOLDER}' for review.")

print(f"\n\n--- BATCH PROCESSING COMPLETE ---")
print(f"Successfully processed {len(os.listdir(OUTPUT_RESULTS_FOLDER))} images.")
print(f"Failed to detect screens in {len(os.listdir(FAILED_FOLDER))} images.")
print(f"Check the '{OUTPUT_RESULTS_FOLDER}' and '{FAILED_FOLDER}' folders for the results.")