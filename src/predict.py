"""Predict a vehicle type for a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from config import CLASS_INDICES_PATH, IMAGE_SIZE, MODEL_PATH, TRAINING_CONFIG_PATH
from utils import load_json, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Предсказание класса транспортного средства по изображению.")
    parser.add_argument("--image", type=Path, required=True, help="Путь к изображению.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Путь к сохраненной модели.")
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
    if not args.image.exists():
        raise FileNotFoundError(f"Изображение не найдено: {args.image}")
    if not args.model.exists():
        raise FileNotFoundError(f"Модель не найдена: {args.model}. Сначала обучите модель.")
    if not CLASS_INDICES_PATH.exists():
        raise FileNotFoundError(f"Словарь классов не найден: {CLASS_INDICES_PATH}. Сначала обучите модель.")

    model = tf.keras.models.load_model(args.model)
    class_indices = load_json(CLASS_INDICES_PATH)

    predicted_class, probabilities = predict_image(
        model,
        args.image,
        class_indices,
        image_size=resolve_image_size(args.image_size),
    )

    print(f"Предсказанный класс: {predicted_class}")
    print("Вероятности:")
    for class_name, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"  {class_name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
