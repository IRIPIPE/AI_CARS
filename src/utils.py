"""Utility functions for dataset handling, model creation and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    SEED,
    SUPPORTED_IMAGE_EXTENSIONS,
    TEST_SIZE,
    VALIDATION_SIZE,
)


PathLike = str | Path


def ensure_parent_dir(path: PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: PathLike) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: PathLike) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def count_images_in_dir(directory: Path) -> int:
    return sum(1 for item in directory.rglob("*") if is_image_file(item))


def has_class_subdirectories(directory: Path) -> bool:
    class_dirs = [item for item in directory.iterdir() if item.is_dir()]
    dirs_with_images = [item for item in class_dirs if count_images_in_dir(item) > 0]
    return len(dirs_with_images) >= 2


def find_dataset_root(raw_data_dir: PathLike) -> Path:
    """Find the directory that directly contains class subdirectories.

    Kaggle archives are not always unpacked in the same shape. This function
    supports both `data/raw/ClassName/*.jpg` and nested layouts such as
    `data/raw/vehicle_images/ClassName/*.jpg`.
    """

    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        raise FileNotFoundError(
            f"Папка с данными не найдена: {raw_data_dir}. "
            "Скачайте и распакуйте датасет в data/raw/."
        )

    if has_class_subdirectories(raw_data_dir):
        return raw_data_dir

    candidates = [
        item
        for item in raw_data_dir.rglob("*")
        if item.is_dir() and has_class_subdirectories(item)
    ]
    if not candidates:
        raise FileNotFoundError(
            "Не удалось найти папку, содержащую подпапки классов с изображениями. "
            "Проверьте структуру data/raw/."
        )

    candidates.sort(key=lambda path: len(path.parts))
    return candidates[0]


def get_class_names(dataset_root: PathLike) -> List[str]:
    dataset_root = Path(dataset_root)
    class_names = [
        item.name
        for item in dataset_root.iterdir()
        if item.is_dir() and count_images_in_dir(item) > 0
    ]
    return sorted(class_names)


def get_image_counts(dataset_root: PathLike) -> Dict[str, int]:
    dataset_root = Path(dataset_root)
    return {class_name: count_images_in_dir(dataset_root / class_name) for class_name in get_class_names(dataset_root)}


def collect_image_paths_and_labels(dataset_root: PathLike) -> Tuple[List[str], List[int], List[str]]:
    dataset_root = Path(dataset_root)
    class_names = get_class_names(dataset_root)
    if len(class_names) < 2:
        raise ValueError("Для обучения требуется минимум два класса изображений.")

    image_paths: List[str] = []
    labels: List[int] = []

    for label, class_name in enumerate(class_names):
        class_dir = dataset_root / class_name
        paths = sorted(str(path) for path in class_dir.rglob("*") if is_image_file(path))
        image_paths.extend(paths)
        labels.extend([label] * len(paths))

    if not image_paths:
        raise ValueError("Изображения не найдены.")

    return image_paths, labels, class_names


def _can_stratify(labels: Sequence[int]) -> bool:
    unique, counts = np.unique(labels, return_counts=True)
    return len(unique) > 1 and np.all(counts >= 3)


def split_dataset(
    image_paths: Sequence[str],
    labels: Sequence[int],
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    seed: int = SEED,
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """Split images into train, validation and test subsets."""

    stratify_labels = labels if _can_stratify(labels) else None
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        list(image_paths),
        list(labels),
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    validation_relative_size = validation_size / (1.0 - test_size)
    stratify_train_val = train_val_labels if _can_stratify(train_val_labels) else None
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=validation_relative_size,
        random_state=seed,
        stratify=stratify_train_val,
    )

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def decode_and_resize_image(path: tf.Tensor, label: tf.Tensor, image_size: Tuple[int, int] = IMAGE_SIZE):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_augmentation_pipeline() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="data_augmentation",
    )


def make_dataset(
    image_paths: Sequence[str],
    labels: Sequence[int],
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    training: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(labels)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: decode_and_resize_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        augmentation = build_augmentation_pipeline()
        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_cnn_model(num_classes: int, image_size: Tuple[int, int] = IMAGE_SIZE) -> tf.keras.Model:
    """Create one transfer-learning CNN based on EfficientNetB0.

    EfficientNetB0 is still a convolutional neural network. The difference from
    a small hand-written CNN is that its convolutional feature extractor is
    pretrained on ImageNet, so it starts with useful visual filters.
    """

    inputs = tf.keras.Input(shape=(*image_size, 3), name="input_image")
    x = tf.keras.layers.Rescaling(255.0, name="restore_0_255_range")(inputs)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
        pooling=None,
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.BatchNormalization(name="classifier_batch_norm")(x)
    x = tf.keras.layers.Dropout(0.35, name="classifier_dropout")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="classifier_dense")(x)
    x = tf.keras.layers.Dropout(0.35, name="classifier_dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="vehicle_type")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vehicle_type_efficientnetb0")


def enable_fine_tuning(model: tf.keras.Model, trainable_layers: int = 40) -> None:
    """Unfreeze the top part of EfficientNetB0 for careful fine-tuning."""

    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True

    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # BatchNormalization statistics are fragile during fine-tuning with small
    # batches, so these layers are kept frozen.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def merge_histories(*histories: tf.keras.callbacks.History) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def plot_training_history(history: tf.keras.callbacks.History | Dict[str, List[float]], output_path: PathLike) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = history.history if hasattr(history, "history") else history
    epochs = range(1, len(metrics.get("loss", [])) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics.get("accuracy", []), label="Train accuracy")
    plt.plot(epochs, metrics.get("val_accuracy", []), label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics.get("loss", []), label="Train loss")
    plt.plot(epochs, metrics.get("val_loss", []), label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def load_and_preprocess_single_image(image_path: PathLike, image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def invert_class_indices(class_indices: Dict[str, int]) -> Dict[int, str]:
    return {int(index): class_name for class_name, index in class_indices.items()}


def predict_image(
    model: tf.keras.Model,
    image_path: PathLike,
    class_indices: Dict[str, int],
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> Tuple[str, Dict[str, float]]:
    image = load_and_preprocess_single_image(image_path, image_size)
    probabilities = model.predict(image, verbose=0)[0]
    index_to_class = invert_class_indices(class_indices)
    predicted_index = int(np.argmax(probabilities))

    probabilities_by_class = {
        index_to_class[index]: float(probabilities[index])
        for index in range(len(probabilities))
    }
    return index_to_class[predicted_index], probabilities_by_class
