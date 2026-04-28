"""Evaluate the saved model and build a classification report and confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    FIGURES_DIR,
    IMAGE_SIZE,
    MODEL_PATH,
    RAW_DATA_DIR,
    TRAINING_CONFIG_PATH,
)
from utils import (
    collect_image_paths_and_labels,
    find_dataset_root,
    load_json,
    make_dataset,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Оценка обученной CNN-модели.")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR, help="Папка с распакованным датасетом.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Путь к сохраненной модели.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Размер batch.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Размер стороны изображения. Если не указан, используется значение из training_config.json.",
    )
    return parser.parse_args()


def resolve_image_size(image_size_arg: int | None) -> tuple[int, int]:
    if image_size_arg is not None:
        return image_size_arg, image_size_arg
    if TRAINING_CONFIG_PATH.exists():
        training_config = load_json(TRAINING_CONFIG_PATH)
        saved_size = training_config.get("image_size", list(IMAGE_SIZE))
        return int(saved_size[0]), int(saved_size[1])
    return IMAGE_SIZE


def main() -> None:
    args = parse_args()
    image_size = resolve_image_size(args.image_size)

    if not args.model.exists():
        raise FileNotFoundError(f"Модель не найдена: {args.model}. Сначала запустите обучение.")
    if not CLASS_INDICES_PATH.exists():
        raise FileNotFoundError(f"Словарь классов не найден: {CLASS_INDICES_PATH}. Сначала запустите обучение.")

    dataset_root = find_dataset_root(args.data_dir)
    image_paths, labels, class_names = collect_image_paths_and_labels(dataset_root)
    _, _, test_paths, _, _, test_labels = split_dataset(image_paths, labels)

    class_indices = load_json(CLASS_INDICES_PATH)
    class_names_by_index = [class_name for class_name, _ in sorted(class_indices.items(), key=lambda item: item[1])]

    test_dataset = make_dataset(test_paths, test_labels, batch_size=args.batch_size, image_size=image_size)
    model = tf.keras.models.load_model(args.model)

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    probabilities = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            predicted_labels,
            target_names=class_names_by_index,
            digits=4,
        )
    )

    matrix = confusion_matrix(test_labels, predicted_labels)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "confusion_matrix.png"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names_by_index,
        yticklabels=class_names_by_index,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Confusion matrix сохранена: {output_path}")


if __name__ == "__main__":
    main()
