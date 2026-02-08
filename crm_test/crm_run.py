import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from paddleocr import PaddleOCR

# -------------------------------
# CONFIG
# -------------------------------
TFLITE_MODEL_PATH = "crm_ssd_mobilenet.tflite"
INPUT_IMAGE_PATH = "crm.jpg"
OUTPUT_IMAGE_PATH = "detection_ocr_results.png"
OUTPUT_JSON_PATH = "ocr_results.json"

CONF_THRESHOLD = 0.5  # SSD detection confidence threshold

# PaddleOCR init (recognition only)
ocr = PaddleOCR(det=False, rec=True, lang='en', use_angle_cls=False, show_log=False)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def preprocess_for_ocr(roi_pil: Image.Image, target_height=48, target_width=320, invert_colors=True):
    """Resize ROI keeping aspect ratio and pad to PaddleOCR input size"""
    # Convert to grayscale
    roi = roi_pil.convert("L")
    if invert_colors:
        roi = Image.eval(roi, lambda x: 255 - x)  # invert colors if needed

    # Compute aspect ratio and resize
    w, h = roi.size
    scale = target_height / h
    new_w = int(w * scale)
    roi = roi.resize((new_w, target_height), Image.Resampling.LANCZOS)

    # Pad to target width
    padded = Image.new("L", (target_width, target_height), color=0)  # black background
    padded.paste(roi, (0, 0))

    # Convert to BGR for OpenCV (PaddleOCR expects 3 channels)
    roi_bgr = cv2.cvtColor(np.array(padded), cv2.COLOR_GRAY2BGR)
    return roi_bgr

# -------------------------------
# LOAD SSD TFLITE MODEL
# -------------------------------
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# LOAD IMAGE
# -------------------------------
if not os.path.exists(INPUT_IMAGE_PATH):
    raise FileNotFoundError(f"{INPUT_IMAGE_PATH} not found")

original_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
orig_w, orig_h = original_image.size
result_image = original_image.copy()
draw = ImageDraw.Draw(result_image)

# Load font
try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

# -------------------------------
# PREPROCESS IMAGE FOR SSD
# -------------------------------
ssd_input_size = input_details[0]['shape'][1]  # typically 300
image_resized = original_image.resize((ssd_input_size, ssd_input_size))
input_data = np.expand_dims(np.array(image_resized, dtype=np.float32) / 255.0, axis=0)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# -------------------------------
# GET DETECTIONS
# -------------------------------
detection_scores = interpreter.get_tensor(output_details[0]['index'])[0]
detection_boxes = interpreter.get_tensor(output_details[1]['index'])[0]
num_detections = int(interpreter.get_tensor(output_details[2]['index'])[0])
detection_classes = interpreter.get_tensor(output_details[3]['index'])[0]

class_names = {1: 'Reading', 2: 'Unit'}  # update as per your model

ocr_results = []

for i in range(num_detections):
    if detection_scores[i] < CONF_THRESHOLD:
        continue

    ymin, xmin, ymax, xmax = detection_boxes[i]
    left = int(xmin * orig_w)
    top = int(ymin * orig_h)
    right = int(xmax * orig_w)
    bottom = int(ymax * orig_h)

    class_id = int(detection_classes[i])
    class_name = class_names.get(class_id, f"Unknown ({class_id})")
    det_conf = float(detection_scores[i])

    # Crop ROI
    roi = original_image.crop((left, top, right, bottom))
    roi_for_ocr = preprocess_for_ocr(roi)

    # Run PaddleOCR
    ocr_result = ocr.ocr(roi_for_ocr, cls=False)
    if ocr_result and ocr_result[0]:
        text, conf = ocr_result[0][0][1]
    else:
        text, conf = "", 0.0

    # Save OCR result
    ocr_results.append({
        "roi_id": i+1,
        "class": class_name,
        "detection_conf": det_conf,
        "ocr_text": text,
        "ocr_conf": float(conf)
    })

    # Draw on image
    draw.rectangle([left, top, right, bottom], outline='red', width=2)
    draw.text((left, top-18), f"{text} ({conf:.0%})", fill="white", font=font)

# -------------------------------
# SAVE RESULTS
# -------------------------------
result_image.save(OUTPUT_IMAGE_PATH)
with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(ocr_results, f, indent=2)

# -------------------------------
# PRINT SUMMARY
# -------------------------------
print("="*60)
print("SSD + PaddleOCR RESULTS")
print("="*60)
for r in ocr_results:
    print(f"ROI #{r['roi_id']} | Class: {r['class']} | DetConf: {r['detection_conf']:.2%} | OCR: {r['ocr_text']} | OCRConf: {r['ocr_conf']:.2%}")
print("="*60)
print(f"Annotated image saved to: {OUTPUT_IMAGE_PATH}")
print(f"OCR results JSON saved to: {OUTPUT_JSON_PATH}")
