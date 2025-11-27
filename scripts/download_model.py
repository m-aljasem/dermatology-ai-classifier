#!/usr/bin/env python3
"""
Download the trained model from GitHub Releases or other sources.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --source releases
    python scripts/download_model.py --source kaggle --path /path/to/model.keras
"""

import argparse
import sys
from pathlib import Path
import urllib.request
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path("models")
MODEL_NAME = "skin_cancer_model.keras"
MODEL_PATH = MODELS_DIR / MODEL_NAME

# GitHub release URL (update this with your actual release URL)
GITHUB_RELEASE_URL = "https://github.com/m-aljasem/dermatology-ai-classifier/releases/latest/download/skin_cancer_model.keras"

def download_from_releases(url: str = None, output_path: Path = None):
    """Download model from GitHub Releases."""
    if url is None:
        url = GITHUB_RELEASE_URL
    if output_path is None:
        output_path = MODEL_PATH
    
    print(f"📥 Downloading model from GitHub Releases...")
    print(f"   URL: {url}")
    print(f"   Destination: {output_path}")
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r   Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end="")
        
        urllib.request.urlretrieve(url, output_path, show_progress)
        print("\n✅ Model downloaded successfully!")
        print(f"   Location: {output_path}")
        print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n💡 Alternative options:")
        print("   1. Download manually from GitHub Releases")
        print("   2. Train your own model using notebooks/02_model_training.ipynb")
        print("   3. Use --source kaggle to copy from Kaggle export")
        return False

def copy_from_kaggle(source_path: str, output_path: Path = None):
    """Copy model from Kaggle export directory."""
    if output_path is None:
        output_path = MODEL_PATH
    
    source = Path(source_path)
    if not source.exists():
        print(f"❌ Source file not found: {source_path}")
        return False
    
    print(f"📋 Copying model from Kaggle export...")
    print(f"   Source: {source}")
    print(f"   Destination: {output_path}")
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        import shutil
        shutil.copy2(source, output_path)
        print(f"✅ Model copied successfully!")
        print(f"   Location: {output_path}")
        print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        return True
    except Exception as e:
        print(f"❌ Error copying model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download or copy the trained model for the Dermatology AI Classifier"
    )
    parser.add_argument(
        "--source",
        choices=["releases", "kaggle"],
        default="releases",
        help="Source to download model from (default: releases)"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to model file (for kaggle source) or custom URL (for releases)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for the model (default: models/skin_cancer_model.keras)"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else MODEL_PATH
    
    # Check if model already exists
    if output_path.exists():
        response = input(f"⚠️  Model already exists at {output_path}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled.")
            return
    
    if args.source == "releases":
        url = args.path if args.path else GITHUB_RELEASE_URL
        success = download_from_releases(url, output_path)
    elif args.source == "kaggle":
        if not args.path:
            print("❌ Error: --path required when using --source kaggle")
            print("   Example: python scripts/download_model.py --source kaggle --path /path/to/best_model.keras")
            return
        success = copy_from_kaggle(args.path, output_path)
    
    if success:
        print("\n🎉 Model is ready! You can now run the Streamlit app:")
        print("   streamlit run app.py")

if __name__ == "__main__":
    main()

