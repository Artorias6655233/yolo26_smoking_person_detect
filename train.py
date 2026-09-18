"""Train a YOLO26 detector on the Smoking_person dataset.

Example:
    python train.py --epochs 150 --batch 32 --device 0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA = Path(__file__).parent / "Smoking_person.v3i.yolo26" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="model config or checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0", help="cuda device, e.g. '0' or 'cpu'")
    parser.add_argument("--patience", type=int, default=50, help="early stopping patience (epochs)")
    parser.add_argument("--project", type=str, default="runs", help="root output dir; final path is <project>/detect/<name>")
    parser.add_argument("--name", type=str, default="smoking_person_yolo26n")
    parser.add_argument("--resume", action="store_true", help="resume from the last checkpoint in --name run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
