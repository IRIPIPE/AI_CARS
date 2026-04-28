"""Train a CNN model for vehicle type classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LAYERS,
    FIGURES_DIR,
    HEAD_EPOCHS,
    IMAGE_SIZE,
    MODEL_PATH,
    MODELS_DIR,
    RAW_DATA_DIR,
    TRAINING_CONFIG_PATH,
)
from utils import (
    build_cnn_model,
    collect_image_paths_and_labels,
    enable_fine_tuning,
    find_dataset_root,
    get_image_counts,
    make_dataset,
    merge_histories,
    plot_training_history,
    save_json,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение CNN для классификации типа транспортного средства.")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR, help="Папка с распакованным датасетом.")
    parser.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS, help="Эпохи обучения классификационной головы.")
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS, help="Эпохи fine-tuning.")
    parser.add_argument("--fine-tune-layers", type=int, default=FINE_TUNE_LAYERS, help="Сколько верхних слоев EfficientNetB0 разморозить.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Размер batch.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE[0], help="Размер стороны изображения.")
    parser.add_argument(
        "--class-weight-mode",
        choices=("none", "sqrt", "balanced"),
        default="none",
        help="Режим весов классов: none обычно лучше для общей accuracy, sqrt мягко балансирует классы.",
    )
    return parser.parse_args()


class BestModelSaver(tf.keras.callbacks.Callback):
    """Save the best model across both training stages by validation accuracy."""

    def __init__(self, filepath: Path, initial_best: float = -np.inf):
        super().__init__()
        self.filepath = filepath
        self.best = initial_best

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        current = logs.get("val_accuracy")
        if current is None:
            return
        if current > self.best:
            self.best = float(current)
            self.model.save(self.filepath)
            print(f"\nval_accuracy improved to {self.best:.4f}; model saved to {self.filepath}")


def build_callbacks(patience: int, initial_best_accuracy: float = -np.inf) -> list[tf.keras.callbacks.Callback]:
    return [
        BestModelSaver(filepath=MODEL_PATH, initial_best=initial_best_accuracy),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(2, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def get_class_weights(labels: list[int], mode: str) -> dict[int, float] | None:
    if mode == "none":
        return None

    classes = np.array(sorted(set(labels)))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(labels))

    if mode == "sqrt":
        weights = np.sqrt(weights)

    return {int(class_index): float(weight) for class_index, weight in zip(classes, weights)}


def main() -> None:
    args = parse_args()
    image_size = (args.image_size, args.image_size)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_root = find_dataset_root(args.data_dir)
    image_paths, labels, class_names = collect_image_paths_and_labels(dataset_root)
    class_indices = {class_name: index for index, class_name in enumerate(class_names)}

    print(f"Найдена папка датасета: {dataset_root}")
    print("Количество изображений по классам:")
    for class_name, count in get_image_counts(dataset_root).items():
        print(f"  {class_name}: {count}")

    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(image_paths, labels)
    print(
        f"Размеры выборок: train={len(train_paths)}, "
        f"validation={len(val_paths)}, test={len(test_paths)}"
    )

    train_dataset = make_dataset(
        train_paths,
        train_labels,
        batch_size=args.batch_size,
        image_size=image_size,
        training=True,
        shuffle=True,
    )
    validation_dataset = make_dataset(val_paths, val_labels, batch_size=args.batch_size, image_size=image_size)
    test_dataset = make_dataset(test_paths, test_labels, batch_size=args.batch_size, image_size=image_size)

    model = build_cnn_model(num_classes=len(class_names), image_size=image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    class_weight = get_class_weights(train_labels, args.class_weight_mode)
    if class_weight is not None:
        print(f"Class weights ({args.class_weight_mode}):")
        for class_name, class_index in class_indices.items():
            print(f"  {class_name}: {class_weight[class_index]:.4f}")
    else:
        print("Class weights: disabled")

    print("\nЭтап 1: обучение классификационной головы EfficientNetB0")
    head_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.head_epochs,
        callbacks=build_callbacks(patience=5),
        class_weight=class_weight,
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    _, best_head_accuracy = model.evaluate(validation_dataset, verbose=0)
    print(f"Лучшая validation accuracy после этапа 1: {best_head_accuracy:.4f}")

    print("\nЭтап 2: fine-tuning верхних слоев EfficientNetB0")
    enable_fine_tuning(model, trainable_layers=args.fine_tune_layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    fine_tune_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.fine_tune_epochs,
        callbacks=build_callbacks(patience=7, initial_best_accuracy=best_head_accuracy),
        class_weight=class_weight,
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    save_json(class_indices, CLASS_INDICES_PATH)
    save_json(
        {
            "dataset_root": str(dataset_root),
            "architecture": "EfficientNetB0 transfer learning",
            "image_size": list(image_size),
            "batch_size": args.batch_size,
            "head_epochs": args.head_epochs,
            "fine_tune_epochs": args.fine_tune_epochs,
            "fine_tune_layers": args.fine_tune_layers,
            "class_weight_mode": args.class_weight_mode,
            "classes": class_names,
            "train_size": len(train_paths),
            "validation_size": len(val_paths),
            "test_size": len(test_paths),
        },
        TRAINING_CONFIG_PATH,
    )

    history_plot_path = FIGURES_DIR / "training_history.png"
    plot_training_history(merge_histories(head_history, fine_tune_history), history_plot_path)

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Словарь классов сохранен: {CLASS_INDICES_PATH}")
    print(f"График обучения сохранен: {history_plot_path}")


if __name__ == "__main__":
    main()
