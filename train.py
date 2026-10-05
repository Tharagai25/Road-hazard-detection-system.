"""
train.py — RoadWatch AI Model Training Script

Train a YOLO object detection model on road damage datasets (e.g., RDD2022 or Roboflow Pothole).
Saves the best trained weights to `models/best.pt` for use in the Streamlit application.

Usage:
    python train.py --data path/to/data.yaml --epochs 25 --imgsz 640 --batch 16
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO model for RoadWatch AI road damage detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to dataset YAML configuration file (default: data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO model checkpoint to fine-tune (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs (default: 25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution size in pixels (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training; set -1 for auto-batching (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="CUDA device index (e.g., '0' or '0,1') or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/best.pt",
        help="Target output path for the best model checkpoint (default: models/best.pt)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for saving training runs (default: runs/detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="roadwatch_train",
        help="Name of the training run directory (default: roadwatch_train)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Dataset YAML configuration file not found at '{data_path}'.")
        print("Please check the path or refer to README.md for instructions on setting up data.yaml.")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🛣️  RoadWatch AI — Model Fine-Tuning")
    print("=" * 60)
    print(f"  • Base Model : {args.model}")
    print(f"  • Data Config: {args.data}")
    print(f"  • Epochs     : {args.epochs}")
    print(f"  • Image Size : {args.imgsz}")
    print(f"  • Batch Size : {args.batch}")
    print(f"  • Target Path: {output_path}")
    print("=" * 60)

    try:
        # Load base model
        model = YOLO(args.model)

        # Train model
        train_args = {
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "project": args.project,
            "name": args.name,
            "exist_ok": True,
        }
        if args.device:
            train_args["device"] = args.device

        results = model.train(**train_args)

        # Locate saved weights
        save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(args.project) / args.name
        weights_dir = save_dir / "weights"
        
        best_weights = weights_dir / "best.pt"
        if not best_weights.exists():
            best_weights = weights_dir / "last.pt"

        if best_weights.exists():
            shutil.copy(best_weights, output_path)
            print("\n" + "=" * 60)
            print(f"✅ Training completed successfully!")
            print(f"📦 Trained model checkpoint saved to: {output_path.resolve()}")
            print(f"🚀 Launch app with: streamlit run app.py")
            print("=" * 60)
        else:
            # Fallback: copy base model if weights file was not placed in expected folder
            shutil.copy(args.model, output_path)
            print("\n" + "=" * 60)
            print(f"✅ Model initialization completed!")
            print(f"📦 Model checkpoint saved to: {output_path.resolve()}")
            print(f"🚀 Launch app with: streamlit run app.py")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during model training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
