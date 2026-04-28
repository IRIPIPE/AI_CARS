"""Check the local Kaggle dataset and show the planned train/val/test split."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR
from utils import collect_image_paths_and_labels, find_dataset_root, get_image_counts, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подготовка и проверка датасета Vehicle Images Dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Путь к папке data/raw/ или папке, содержащей подпапки классов.",
    )
    return parser.parse_args()


def print_split_distribution(name: str, labels: list[int], class_names: list[str]) -> None:
    counts = Counter(labels)
    print(f"\n{name}: {len(labels)} изображений")
    for index, class_name in enumerate(class_names):
        print(f"  {class_name}: {counts.get(index, 0)}")


def main() -> None:
    args = parse_args()
    dataset_root = find_dataset_root(args.data_dir)
    image_paths, labels, class_names = collect_image_paths_and_labels(dataset_root)
    counts = get_image_counts(dataset_root)

    print(f"Найдена папка датасета: {dataset_root}")
    print("\nКлассы и количество изображений:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")

    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(image_paths, labels)

    print_split_distribution("Train", train_labels, class_names)
    print_split_distribution("Validation", val_labels, class_names)
    print_split_distribution("Test", test_labels, class_names)

    preview = pd.DataFrame(
        {
            "path": train_paths[:5] + val_paths[:5] + test_paths[:5],
            "split": ["train"] * min(5, len(train_paths))
            + ["validation"] * min(5, len(val_paths))
            + ["test"] * min(5, len(test_paths)),
        }
    )
    if not preview.empty:
        print("\nПример записей после разделения:")
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
