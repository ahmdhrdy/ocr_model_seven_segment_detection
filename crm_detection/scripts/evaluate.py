"""
Evaluation Script for CRM Detection Model
Validates trained model on validation dataset
"""

import os
import sys
import subprocess

# Add TensorFlow models and models/research to Python path
models_path = os.path.join(os.path.dirname(__file__), "../models/research")
models_official_path = os.path.join(os.path.dirname(__file__), "../models")
if models_path not in sys.path:
    sys.path.insert(0, models_path)
if models_official_path not in sys.path:
    sys.path.insert(0, models_official_path)

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util

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
            print("⚠️  No GPU detected. Evaluation will use CPU (slower)")
        
        # Check if Object Detection API is available
        from object_detection.utils import config_util
        print("✅ TensorFlow Object Detection API found")
        return True
    except ImportError as e:
        print(f"❌ Error: {e}")
        return False

def evaluate_model():
    """Run evaluation on validation dataset."""
    
    print("\n" + "="*70)
    print("EVALUATING TRAINED MODEL")
    print("="*70 + "\n")
    
    # Set up environment with proper PYTHONPATH
    env = os.environ.copy()
    models_path = os.path.join(os.path.dirname(__file__), "../models/research")
    models_official_path = os.path.join(os.path.dirname(__file__), "../models")
    
    pythonpath = f"{models_path}:{models_official_path}"
    if 'PYTHONPATH' in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env['PYTHONPATH'] = pythonpath
    
    # Evaluation command
    cmd = [
        "python",
        "-m", "object_detection.model_main_tf2",
        "--model_dir=training/",
        "--pipeline_config_path=pipeline.config",
        "--checkpoint_dir=training/"
    ]
    
    try:
        print("Starting evaluation on validation dataset...")
        print("This may take a few minutes...\n")
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(0)

def display_metrics():
    """Display evaluation metrics from training logs."""
    
    print("\n" + "="*70)
    print("VALIDATION METRICS")
    print("="*70)
    
    # Try to read metrics from latest checkpoint
    checkpoint_dir = "training/"
    if os.path.exists(checkpoint_dir):
        print(f"\n✅ Training directory found: {checkpoint_dir}")
        
        # List checkpoint files
        checkpoint_files = []
        for file in os.listdir(checkpoint_dir):
            if file.startswith('ckpt-') and file.endswith('.index'):
                checkpoint_num = file.replace('ckpt-', '').replace('.index', '')
                checkpoint_files.append(int(checkpoint_num))
        
        if checkpoint_files:
            latest_ckpt = max(checkpoint_files)
            print(f"✅ Latest checkpoint: ckpt-{latest_ckpt}")
        
        # Check for event files (TensorBoard logs)
        train_dir = os.path.join(checkpoint_dir, "train")
        if os.path.exists(train_dir):
            print(f"✅ TensorBoard logs found in {train_dir}")
            print("\n   View metrics with:")
            print("   tensorboard --logdir=training/")
    else:
        print(f"\n❌ Training directory not found: {checkpoint_dir}")
        print("Please train the model first: python scripts/train.py")

def main():
    """Main evaluation workflow."""
    
    print("="*70)
    print("CRM DETECTION MODEL - EVALUATION")
    print("="*70 + "\n")
    
    # Step 1: Check environment
    print("Step 1: Checking environment...")
    if not check_environment():
        sys.exit(1)
    print()
    
    # Step 2: Display current metrics
    print("Step 2: Checking available metrics...")
    display_metrics()
    print()
    
    # Step 3: Run evaluation
    print("Step 3: Running evaluation...")
    response = input("Start evaluation? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Evaluation cancelled.")
        sys.exit(0)
    
    evaluate_model()
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETED")
    print("="*70)
    print("\nNext steps:")
    print("1. View detailed metrics: tensorboard --logdir=training/")
    print("2. Export model: python scripts/export_tflite.py")

if __name__ == "__main__":
    main()
