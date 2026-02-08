# CRM OCR Detection Pipeline (CPU / Cross-Platform)

## Overview

This project performs **end-to-end OCR** on images using:

1. **TensorFlow Lite SSD model** for ROI (Region of Interest) detection
2. **PaddleOCR** for text & numeric recognition
3. **CPU-only runtime** (Windows / Linux / Android compatible)

It is designed to extract structured values such as:
- R-RES / Y-RES / B-RES
- Numeric readings (e.g. 09.38, 10.26, 04.24)

---

## Pipeline Architecture

```
Input Image
    ↓
TensorFlow Lite SSD (ROI Detection)
    ↓
ROI Cropping + Preprocessing
    ↓
PaddleOCR (Text Recognition)
    ↓
JSON Output
```

---

## Supported Platforms

| Platform | Supported |
|----------|-----------|
| Windows (CPU) | ✅ |
| Linux (CPU) | ✅ |
| Android (CPU / NDK) | ✅ |
| GPU | ❌ Not required |

---

## Requirements

- Python **3.9**
- CPU only (no CUDA needed)

---

## Installation (One-time)

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / Android shell:**
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Scripts

```bash
python crm_run.py
```

### Output

**Console:**
```
ROI #1 | Class: Reading | DetConf: 99.47% | OCR: 04.24 | OCRConf: 99.07%
```

**Generated Files:**
- `detection_ocr_results.png` → output image
- `ocr_results.json` → Structured OCR output

---

## OCR Model Details

| Parameter | Value |
|-----------|-------|
| **OCR Engine** | PaddleOCR |
| **Recognition Model** | PP-OCRv5 mobile |
| **Language** | English |
| **Input** | Cropped ROI images |
| **Output** | Recognized text + Confidence score (0–1) |

---

## Android Deployment Notes

- TensorFlow Lite model can be used directly
- PaddleOCR requires:
  - Paddle Lite OR
  - Server inference fallback

---

## Troubleshooting

### ModuleNotFoundError: paddle

```bash
pip install paddlepaddle==2.6.2
```

---

## `crm_run.py` Scripts Pipeline

```
crm_run.py
│
├─ TensorFlow Lite SSD model (crm_ssd_mobilenet.tflite - Detection Model)
│   └─ Detects 6 ROIs (boxes)
|
|-Paddle Lite model (en_PP-OCRv4_rec_lite.nb - Recognition Model for android deployment)
│
├─ ROI preprocessing (OpenCV / NumPy)
│   └─ Crop + resize + clean background
│
├─ PaddleOCR (Python API)
│   ├─ Text detection (disabled / bypassed)
│   └─ Text recognition (ACTIVE)
│
└─ Final output
    ├─ OCR text + confidence
    ├─ output image
    └─ JSON
```

---

## How to "GET" the PaddleOCR Model

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_gpu=False)
```

### What PaddleOCR Does Internally

When `PaddleOCR()` runs:

1. It checks: `~/.paddleocr/`
2. If models are missing → it downloads them automatically
3. It selects default models based on:
   - `language = "en"`
   - `device = CPU`
   - `mode = inference`

**Folder where the model files will exist:**
```
~/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer/
```

**Files:**
- `inference.pdmodel` - network structure
- `inference.pdiparams` - weights
- `inference.pdiparams.info`

---

## To Inspect PaddleOCR Model Details

### 1. Print Model Architecture

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_gpu=False)
print(ocr.text_recognizer.model)
```

### 2. Print Input Tensor Shape

```python
model = ocr.text_recognizer.model
for inp in model.inputs():
    print(inp.name, inp.shape, inp.dtype)
```

### 3. Print Output Tensor Shape

```python
for out in model.outputs():
    print(out.name, out.shape, out.dtype)
```

### 4. Enable Full Inference Logs

```python
ocr = PaddleOCR(use_gpu=False, show_log=True)
```