import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models import MODEL_IMAGE_SIZE, OUTPUT_CLASSES, RPSClassifier

CLASS_TO_TARGET = {
    "P": [1.0, 0.0, 0.0],
    "R": [0.0, 1.0, 0.0],
    "S": [0.0, 0.0, 1.0],
    "N": [0.0, 0.0, 0.0],
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3-output Rock/Paper/Scissors confidence model with negative samples"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/mnt/hdd16t0/dataset/rps_dataset/processed_dataset",
        help="Dataset root dir with N/P/R/S subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/rps_mobilenetv1",
        help="Directory to save checkpoints and metadata",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for each class",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader worker count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold used by validation metrics",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load pretrained MobileNetV1 weights from timm",
    )
    parser.add_argument(
        "--head_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the custom classification head",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout used in the custom classification head",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_samples(dataset_dir: Path):
    samples_by_class = defaultdict(list)
    for class_name, target in CLASS_TO_TARGET.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples_by_class[class_name].append((path, torch.tensor(target, dtype=torch.float32)))
    return samples_by_class


def split_samples(samples_by_class, val_ratio: float, seed: int):
    rng = random.Random(seed)
    train_samples = []
    val_samples = []

    for class_name, items in samples_by_class.items():
        items = items.copy()
        rng.shuffle(items)
        if not items:
            continue

        val_count = int(len(items) * val_ratio)
        if len(items) > 1:
            val_count = max(1, val_count)
            val_count = min(len(items) - 1, val_count)
        else:
            val_count = 0

        val_samples.extend(items[:val_count])
        train_samples.extend(items[val_count:])
        print(
            f"{class_name}: total={len(items)} train={len(items[val_count:])} val={len(items[:val_count])}"
        )

    return train_samples, val_samples


class RPSDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, target, str(image_path)


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(
                image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=18),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.0)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def build_model(
    pretrained: bool, image_size: int, head_hidden_dim: int, dropout: float
):
    return RPSClassifier(
        image_size=image_size,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
        num_classes=len(OUTPUT_CLASSES),
        pretrained=pretrained,
    )


def multilabel_exact_match(probs: torch.Tensor, targets: torch.Tensor, threshold: float):
    preds = (probs >= threshold).float()
    return (preds == targets).all(dim=1).float().mean().item()


def negative_recall(probs: torch.Tensor, targets: torch.Tensor, threshold: float):
    negative_mask = targets.sum(dim=1) == 0
    if negative_mask.sum() == 0:
        return 0.0
    preds = (probs[negative_mask] >= threshold).float()
    true_negative = (preds.sum(dim=1) == 0).float().mean().item()
    return true_negative


def positive_top1_accuracy(probs: torch.Tensor, targets: torch.Tensor):
    positive_mask = targets.sum(dim=1) > 0
    if positive_mask.sum() == 0:
        return 0.0
    pred_idx = probs[positive_mask].argmax(dim=1)
    target_idx = targets[positive_mask].argmax(dim=1)
    return (pred_idx == target_idx).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, threshold):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0
    all_probs = []
    all_targets = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        all_probs.append(torch.sigmoid(logits).detach().cpu())
        all_targets.append(targets.detach().cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = {
        "loss": total_loss / max(total_items, 1),
        "exact_match": multilabel_exact_match(probs, targets, threshold),
        "positive_top1": positive_top1_accuracy(probs, targets),
        "negative_recall": negative_recall(probs, targets, threshold),
    }
    return metrics


def save_metadata(output_dir: Path, args, train_count: int, val_count: int):
    metadata = {
        "model_name": "mobilenetv1_100",
        "output_classes": OUTPUT_CLASSES,
        "negative_class_dir": "N",
        "image_size": MODEL_IMAGE_SIZE,
        "dataset_dir": args.dataset_dir,
        "head_hidden_dim": args.head_hidden_dim,
        "dropout": args.dropout,
        "train_count": train_count,
        "val_count": val_count,
        "threshold": args.threshold,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise RuntimeError(f"dataset directory not found: {dataset_dir}")

    samples_by_class = list_samples(dataset_dir)
    missing = [class_name for class_name in ["P", "R", "S"] if not samples_by_class.get(class_name)]
    if missing:
        raise RuntimeError(f"missing required class data: {missing}")
    if not samples_by_class.get("N"):
        print("Warning: class N not found, training will proceed without negative samples.")

    train_samples, val_samples = split_samples(
        samples_by_class=samples_by_class,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if not train_samples or not val_samples:
        raise RuntimeError("train/val split failed, please check dataset size and val_ratio")

    train_transform, val_transform = build_transforms(MODEL_IMAGE_SIZE)
    train_dataset = RPSDataset(train_samples, train_transform)
    val_dataset = RPSDataset(val_samples, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        pretrained=args.pretrained,
        image_size=MODEL_IMAGE_SIZE,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_metric = -1.0
    best_checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"

    save_metadata(output_dir, args, len(train_dataset), len(val_dataset))

    print(f"Device      : {device}")
    print(f"Image size  : {MODEL_IMAGE_SIZE}")
    print(f"Train count : {len(train_dataset)}")
    print(f"Val count   : {len(val_dataset)}")
    print(f"Output dir  : {output_dir}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=args.threshold,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            threshold=args.threshold,
        )
        scheduler.step()

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "image_size": MODEL_IMAGE_SIZE,
            "output_classes": OUTPUT_CLASSES,
            "threshold": args.threshold,
            "head_hidden_dim": args.head_hidden_dim,
            "dropout": args.dropout,
        }
        torch.save(checkpoint, last_checkpoint_path)

        score = (val_metrics["positive_top1"] + val_metrics["negative_recall"]) / 2.0
        if score > best_metric:
            best_metric = score
            torch.save(checkpoint, best_checkpoint_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_exact={val_metrics['exact_match']:.4f} "
            f"val_pos_top1={val_metrics['positive_top1']:.4f} "
            f"val_neg_recall={val_metrics['negative_recall']:.4f}"
        )

    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")


if __name__ == "__main__":
    main()
