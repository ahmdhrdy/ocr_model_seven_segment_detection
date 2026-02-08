"""
Training Script for CRM Detection Model
SSD MobileNet V2 - Transfer Learning
"""

import os
import sys
import subprocess

# Add TensorFlow models and models/research to Python path for Object Detection API
models_path = os.path.join(os.path.dirname(__file__), "../models/research")
models_official_path = os.path.join(os.path.dirname(__file__), "../models")
if models_path not in sys.path:
    sys.path.insert(0, models_path)
if models_official_path not in sys.path:
    sys.path.insert(0, models_official_path)

def check_environment():
    """Verify TensorFlow Object Detection API is installed."""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} found")
                # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} GPU(s) available")
            for gpu in gpus:
                print(f"   - {gpu}")
            # Configure GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
                # Check if Object Detection API is available
        from object_detection.utils import config_util
        print("✅ TensorFlow Object Detection API found")
        return True
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("\nPlease install TensorFlow Object Detection API:")
        print("git clone https://github.com/tensorflow/models.git")
        print("cd models/research")
        print("protoc object_detection/protos/*.proto --python-out=.")
        print("pip install .")
        return False

def verify_files():
    """Check if all required files exist."""
    required_files = [
        "annotations/label_map.pbtxt",
        "tfrecords/train.record",
        "tfrecords/val.record",
        "pipeline.config"
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
        else:
            print(f"✅ Found: {file}")
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
        return False
    
    return True

def check_checkpoint():
    """Verify pre-trained checkpoint exists."""
    checkpoint_path = "pre_trained_model/ssd_mobilenet_v2_320x320/checkpoint/ckpt-0.index"
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Pre-trained checkpoint found")
        return True
    else:
        print(f"⚠️  Warning: Pre-trained checkpoint not found at {checkpoint_path}")
        print("Training will start from scratch (not recommended)")
        return False

def train_model():
    """Start training process."""
    
    print("\n" + "="*70)
    print("STARTING TRAINING - CRM DETECTION MODEL")
    print("="*70 + "\n")
    
    # Set up environment with proper PYTHONPATH
    env = os.environ.copy()
    models_path = os.path.join(os.path.dirname(__file__), "../models/research")
    models_official_path = os.path.join(os.path.dirname(__file__), "../models")
    
    # Build PYTHONPATH
    pythonpath = f"{models_path}:{models_official_path}"
    if 'PYTHONPATH' in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env['PYTHONPATH'] = pythonpath
    
    # Training command
    cmd = [
        "python",
        "-m", "object_detection.model_main_tf2",
        "--model_dir=training/",
        "--pipeline_config_path=pipeline.config",
        "--num_train_steps=8000",
        "--sample_1_of_n_eval_examples=1",
        "--alsologtostderr"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)

def main():
    """Main training workflow."""
    
    print("="*70)
    print("CRM DETECTION MODEL - TRAINING SETUP")
    print("="*70 + "\n")
    
    # Step 1: Check environment
    print("Step 1: Checking environment...")
    if not check_environment():
        sys.exit(1)
    print()
    
    # Step 2: Verify files
    print("Step 2: Verifying project files...")
    if not verify_files():
        sys.exit(1)
    print()
    
    # Step 3: Check checkpoint
    print("Step 3: Checking pre-trained checkpoint...")
    check_checkpoint()
    print()
    
    # Step 4: Start training
    print("Step 4: Starting training...")
    print("\n📊 Training Configuration:")
    print("   - Model: SSD MobileNet V2")
    print("   - Input Size: 320x320")
    print("   - Classes: 2 (Mode, Reading)")
    print("   - Training Steps: 8,000")
    print("   - Batch Size: 16")
    print("   - Transfer Learning: Enabled")
    print("\n⏱️  Estimated Time: 4-8 hours (GPU) / 24+ hours (CPU)\n")
    
    # Confirm start
    response = input("Start training? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Training cancelled.")
        sys.exit(0)
    
    train_model()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: tensorboard --logdir=training/")
    print("2. Export model: python scripts/export_tflite.py")

if __name__ == "__main__":
    main()