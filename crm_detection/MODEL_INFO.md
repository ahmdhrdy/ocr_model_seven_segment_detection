# CRM Detection Model - SSD MobileNet V2
    
## Model Information
- Architecture: SSD MobileNet V2
- Input Size: 320x320x3
- Output Format: 4 tensors (boxes, classes, scores, num_detections)
- Classes: 2 (Mode, Reading)
- Quantization: INT8 (Default Optimization)

## Output Tensors
1. Boxes: [1, 10, 4] - Bounding box coordinates [ymin, xmin, ymax, xmax] normalized [0-1]
2. Classes: [1, 10] - Class indices (1=Mode, 2=Reading)
3. Scores: [1, 10] - Confidence scores [0-1]
4. Num Detections: [1] - Number of valid detections (0-10)

## Usage
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
interpreter = tf.lite.Interpreter(model_path='crm_ssd_mobilenet.tflite')
interpreter.allocate_tensors()

# Load image
image = Image.open('test.jpg').resize((320, 320))
input_data = np.expand_dims(np.array(image, dtype=np.uint8), axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get outputs
boxes = interpreter.get_tensor(output_details[0]['index'])[0]
classes = interpreter.get_tensor(output_details[1]['index'])[0]
scores = interpreter.get_tensor(output_details[2]['index'])[0]
num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
```

## Performance
- Inference Time: <100ms (target)
- Model Size: ~4MB
- Accuracy: TBD (depends on training)
