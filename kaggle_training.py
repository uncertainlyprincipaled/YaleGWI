#!/usr/bin/env python3
"""
Kaggle Training Script

This script is optimized for training models on Kaggle's free GPU environment.
It handles the less computationally intensive models:
- SpecProj-UNet (baseline)
- Heat-Kernel Model

Usage:
    python kaggle_training.py --model specproj_unet --epochs 30
    python kaggle_training.py --model heat_kernel --epochs 30
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append('/kaggle/working/YaleGWI/src')

def setup_kaggle_environment():
    """Set up Kaggle environment for training."""
    print("🚀 Setting up Kaggle environment...")
    
    # Set environment variables
    os.environ['GWI_ENV'] = 'kaggle'
    os.environ['DEBUG_MODE'] = '0'
    
    # Verify GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️ No GPU available - will use CPU")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    return True

def train_model(model_type, epochs=30, batch_size=16, use_amp=True):
    """Train a specific model on Kaggle."""
    print(f"🎯 Training {model_type} for {epochs} epochs...")
    
    try:
        from src.core.train import train
        from src.core.config import CFG
        
        # Configure for Kaggle
        CFG.batch = batch_size
        CFG.epochs = epochs
        CFG.use_amp = use_amp
        
        # Start training
        start_time = time.time()
        train(model_type=model_type, fp16=use_amp)
        end_time = time.time()
        
        training_time = (end_time - start_time) / 3600  # hours
        print(f"✅ Training completed in {training_time:.2f} hours")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def export_model(model_type, upload_to_s3=True):
    """Export trained model to S3."""
    print(f"📤 Exporting {model_type} model...")
    
    try:
        from src.core.train import export_model
        
        # Export to S3
        if upload_to_s3:
            success = export_model(model_type, upload_to_s3=True)
            if success:
                print(f"✅ {model_type} model exported to S3")
                return True
            else:
                print(f"❌ Failed to export {model_type} model to S3")
                return False
        else:
            print(f"✅ {model_type} model saved locally")
            return True
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def keep_alive():
    """Keep the runtime alive during long training."""
    print("💡 Keep-alive script started. Training will continue...")
    while True:
        time.sleep(300)  # Print every 5 minutes
        print(f"⏰ Still training... {time.strftime('%H:%M:%S')}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train models on Kaggle")
    parser.add_argument('--model', type=str, required=True, 
                       choices=['specproj_unet', 'heat_kernel'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    parser.add_argument('--keep-alive', action='store_true',
                       help='Run keep-alive script in background')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"🎯 Kaggle Training: {args.model}")
    print("="*60)
    
    # Setup environment
    if not setup_kaggle_environment():
        print("❌ Environment setup failed")
        return
    
    # Start keep-alive if requested
    if args.keep_alive:
        import threading
        keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
        keep_alive_thread.start()
        print("✅ Keep-alive script started in background")
    
    # Train model
    success = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_amp=not args.no_amp
    )
    
    if success:
        # Export model
        export_success = export_model(args.model, upload_to_s3=True)
        
        if export_success:
            print("🎉 Training and export completed successfully!")
            print(f"📁 Model saved as: outputs/{args.model}_best.pth")
            print(f"☁️ Model uploaded to S3: s3://yale-gwi/models/{args.model}_best.pth")
        else:
            print("⚠️ Training completed but export failed")
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    main() 