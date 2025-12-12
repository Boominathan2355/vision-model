#!/usr/bin/env python
"""
Master Training Script for VelCore Trillion Parameter Model
Downloads full datasets and trains models with optimized settings
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def check_dependencies():
    """Check if all required packages are installed"""
    print_header("CHECKING DEPENDENCIES")
    
    required_packages = [
        'torch',
        'torchvision',
        'datasets',
        'transformers',
        'tqdm',
        'pillow',
        'numpy',
        'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True

def check_system_info():
    """Check system information"""
    print_header("SYSTEM INFORMATION")
    
    import torch
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  ⚠️  Running on CPU (training will be slow)")
    
    print(f"  Working Directory: {os.getcwd()}")

def list_training_scripts():
    """List available training scripts"""
    print_header("AVAILABLE TRAINING SCRIPTS")
    
    scripts = {
        '1': {
            'name': 'train_final.py',
            'dataset': 'Turing-Open-Reasoning',
            'samples': '~1000',
            'description': 'Multi-domain reasoning problems'
        },
        '2': {
            'name': 'train_gsm8k.py',
            'dataset': 'OpenAI GSM8K',
            'samples': '~7500',
            'description': 'Grade school math word problems'
        },
        '3': {
            'name': 'train_geometry.py',
            'dataset': 'Geometry3K',
            'samples': '~2100',
            'description': 'Geometry with REAL diagram images'
        },
        '4': {
            'name': 'train_socratic.py',
            'dataset': 'GSM8K (Socratic)',
            'samples': '~7500',
            'description': 'Socratic method step-by-step reasoning'
        },
        '5': {
            'name': 'train_conversation.py',
            'dataset': 'Conversation Data',
            'samples': 'Variable',
            'description': 'Multi-turn dialogue training'
        },
        'all': {
            'name': 'ALL SCRIPTS',
            'dataset': 'All Datasets',
            'samples': '~18000+',
            'description': 'Sequential training on all datasets'
        }
    }
    
    for key, info in scripts.items():
        print(f"\n  [{key}] {info['name']}")
        print(f"      Dataset: {info['dataset']}")
        print(f"      Samples: {info['samples']}")
        print(f"      Purpose: {info['description']}")
    
    return scripts

def run_training_script(script_name):
    """Run a training script"""
    print_header(f"STARTING TRAINING: {script_name}")
    
    script_path = Path(f"d:\\Model\\{script_name}")
    
    if not script_path.exists():
        print(f"❌ Error: {script_name} not found!")
        return False
    
    try:
        print(f"Executing: python {script_name}\n")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd="d:\\Model",
            capture_output=False
        )
        
        if result.returncode == 0:
            print_header(f"✓ {script_name} COMPLETED SUCCESSFULLY")
            return True
        else:
            print_header(f"✗ {script_name} FAILED")
            return False
    
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def run_sequential_training():
    """Run all training scripts sequentially"""
    print_header("SEQUENTIAL TRAINING: ALL DATASETS")
    
    scripts = [
        'train_final.py',
        'train_gsm8k.py',
        'train_geometry.py',
        'train_socratic.py',
        'train_conversation.py'
    ]
    
    results = {}
    
    for i, script in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] Running {script}...")
        results[script] = run_training_script(script)
        
        if not results[script]:
            print(f"⚠️  {script} had issues, continuing to next...")
    
    # Summary
    print_header("TRAINING SUMMARY")
    
    for script, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {script}")
    
    successful = sum(1 for v in results.values() if v)
    print(f"\nCompleted: {successful}/{len(scripts)} scripts")

def main():
    """Main entry point"""
    print_header("VELCORE TRAINING LAUNCHER")
    print("  Trillion Parameter Vision-Language Model")
    print("  GPU-Optimized Multi-Dataset Training")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # System info
    check_system_info()
    
    # List scripts
    scripts = list_training_scripts()
    
    # Ask user which script to run
    print_header("SELECT TRAINING MODE")
    print("\nEnter your choice:")
    choice = input("  > ").strip()
    
    if choice == 'all':
        run_sequential_training()
    elif choice in scripts:
        script_name = scripts[choice]['name']
        success = run_training_script(script_name)
        
        if success:
            print_header("✓ TRAINING COMPLETED SUCCESSFULLY")
            print("\nNext steps:")
            print("  1. Check training_history.json for loss curves")
            print("  2. Evaluate model with: python evaluate_model.py")
            print("  3. Run inference with: python ask_model.py")
        
        return 0 if success else 1
    else:
        print("❌ Invalid choice. Please run again and select 1-5 or 'all'")
        return 1

if __name__ == "__main__":
    sys.exit(main())
