"""
Export Script for CRM Detection Model
Converts trained model to TFLite with 4 output tensors
Compatible with Android ObjectDetector
"""

import os
import sys

# Add TensorFlow models and models/research to Python path
models_path = os.path.join(os.path.dirname(__file__), "../models/research")
models_official_path = os.path.join(os.path.dirname(__file__), "../models")
if models_path not in sys.path:
    sys.path.insert(0, models_path)
if models_official_path not in sys.path:
    sys.path.insert(0, models_official_path)

import tensorflow as tf

def export_savedmodel():
    """Export checkpoint to SavedModel format."""
    
    print("\n" + "="*70)
    print("STEP 1: EXPORTING TO SAVEDMODEL FORMAT")
    print("="*70 + "\n")
    
    try:
        from object_detection import export_tflite_graph_lib_tf2
        from object_detection.utils import config_util
        from object_detection.protos import pipeline_pb2
    except ImportError as e:
        print(f"❌ TensorFlow Object Detection API not found: {e}")
        sys.exit(1)
    
    pipeline_config_path = 'pipeline.config'
    trained_checkpoint_dir = 'training/'
    output_directory = 'exported_model/'
    
    # Check if training checkpoint exists
    if not os.path.exists(os.path.join(trained_checkpoint_dir, 'checkpoint')):
        print(f"❌ No checkpoint found in {trained_checkpoint_dir}")
        print("Please train the model first: python scripts/train.py")
        sys.exit(1)
    
    # Parse pipeline config and reconstruct as proto
    print("Parsing pipeline config...")
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    
    # Create the pipeline config proto with all components
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_config.model.CopyFrom(configs['model'])
    pipeline_config.train_config.CopyFrom(configs['train_config'])
    
    # Export TFLite-compatible model
    print("Exporting TFLite-compatible model...")
    export_tflite_graph_lib_tf2.export_tflite_model(
        pipeline_config=pipeline_config,
        trained_checkpoint_dir=trained_checkpoint_dir,
        output_directory=output_directory,
        max_detections=10,
        use_regular_nms=True
    )
    
    print(f"✅ Model exported to {output_directory}")
    return output_directory

def convert_to_tflite(saved_model_dir):
    """Convert SavedModel to TFLite format."""
    
    print("\n" + "="*70)
    print("STEP 2: CONVERTING TO TFLITE FORMAT")
    print("="*70 + "\n")
    
    saved_model_path = os.path.join(saved_model_dir, 'saved_model')
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # Set optimization (INT8 quantization for smaller size)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Ensure compatibility with ObjectDetector
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    
    # Convert
    print("Converting to TFLite (this may take a few minutes)...")
    tflite_model = converter.convert()
    
    # Save model
    output_path = 'crm_ssd_mobilenet.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ TFLite model saved: {output_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    
    return output_path

def verify_tflite_model(model_path):
    """Verify the TFLite model has correct output structure."""
    
    print("\n" + "="*70)
    print("STEP 3: VERIFYING MODEL STRUCTURE")
    print("="*70 + "\n")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print input details
    print("📥 INPUT TENSOR:")
    for i, detail in enumerate(input_details):
        print(f"   Index {i}:")
        print(f"      Shape: {detail['shape']}")
        print(f"      Type: {detail['dtype']}")
        print(f"      Name: {detail['name']}")
    
    # Print output details
    print("\n📤 OUTPUT TENSORS:")
    for i, detail in enumerate(output_details):
        print(f"   Index {i}:")
        print(f"      Shape: {detail['shape']}")
        print(f"      Type: {detail['dtype']}")
        print(f"      Name: {detail['name']}")
    
    # Verify output structure
    print("\n🔍 VERIFICATION:")
    
    if len(output_details) != 4:
        print(f"   ❌ Expected 4 output tensors, got {len(output_details)}")
        print("   ⚠️  Model may not be compatible with ObjectDetector")
        return False
    
    expected_shapes = [
        (1, 10, 4),   # Boxes [batch, num_detections, 4]
        (1, 10),      # Classes [batch, num_detections]
        (1, 10),      # Scores [batch, num_detections]
        (1,)          # Number of detections [batch]
    ]
    
    all_correct = True
    for i, (detail, expected) in enumerate(zip(output_details, expected_shapes)):
        actual = tuple(detail['shape'])
        if actual == expected:
            print(f"   ✅ Output {i}: {actual} - CORRECT")
        else:
            print(f"   ❌ Output {i}: {actual} - Expected {expected}")
            all_correct = False
    
    if all_correct:
        print("\n✅ MODEL STRUCTURE VERIFIED - Compatible with ObjectDetector!")
    else:
        print("\n⚠️  MODEL STRUCTURE MISMATCH - May need re-export")
    
    return all_correct

def create_metadata():
    """Create model metadata file."""
    
    metadata = f"""# CRM Detection Model - SSD MobileNet V2
    
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
"""
    
    with open('MODEL_INFO.md', 'w') as f:
        f.write(metadata)
    
    print("\n📄 Model metadata saved: MODEL_INFO.md")

def main():
    """Main export workflow."""
    
    print("="*70)
    print("CRM DETECTION MODEL - TFLITE EXPORT")
    print("="*70)
    
    # Step 1: Export to SavedModel
    saved_model_dir = export_savedmodel()
    
    # Step 2: Convert to TFLite
    tflite_path = convert_to_tflite(saved_model_dir)
    
    # Step 3: Verify structure
    is_valid = verify_tflite_model(tflite_path)
    
    # Step 4: Create metadata
    create_metadata()
    
    print("\n" + "="*70)
    print("✅ EXPORT COMPLETED")
    print("="*70)
    print(f"\n📦 Final Model: {tflite_path}")
    print(f"📄 Documentation: MODEL_INFO.md")
    
    if is_valid:
        print("\n✅ Model is ready for deployment!")
        print("\nNext steps:")
        print("1. Test the model: python scripts/test_tflite.py")
        print("2. Deliver to client: crm_ssd_mobilenet.tflite + label_map.pbtxt")
    else:
        print("\n⚠️  Model structure needs verification")
        print("Please check the output tensor shapes")

if __name__ == "__main__":
    main()