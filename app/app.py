"""Flask interface for vehicle type prediction."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import tensorflow as tf
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import CLASS_INDICES_PATH, IMAGE_SIZE, MODEL_PATH, TRAINING_CONFIG_PATH, UPLOAD_DIR
from utils import load_json, predict_image


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

model = None
class_indices = None
inference_image_size = IMAGE_SIZE


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_resources() -> None:
    global model, class_indices, inference_image_size
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}. Сначала запустите обучение.")
        model = tf.keras.models.load_model(MODEL_PATH)
    if class_indices is None:
        if not CLASS_INDICES_PATH.exists():
            raise FileNotFoundError(f"Словарь классов не найден: {CLASS_INDICES_PATH}. Сначала запустите обучение.")
        class_indices = load_json(CLASS_INDICES_PATH)
    if TRAINING_CONFIG_PATH.exists():
        training_config = load_json(TRAINING_CONFIG_PATH)
        saved_size = training_config.get("image_size", list(IMAGE_SIZE))
        inference_image_size = (int(saved_size[0]), int(saved_size[1]))


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None
    image_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            error = "Выберите изображение для загрузки."
        elif not allowed_file(file.filename):
            error = "Поддерживаются форматы: jpg, jpeg, png, bmp, webp."
        else:
            try:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                filename = secure_filename(file.filename)
                extension = filename.rsplit(".", 1)[1].lower()
                saved_name = f"{uuid4().hex}.{extension}"
                saved_path = UPLOAD_DIR / saved_name
                file.save(saved_path)

                load_resources()
                prediction, probabilities_raw = predict_image(
                    model,
                    saved_path,
                    class_indices,
                    image_size=inference_image_size,
                )
                probabilities = sorted(
                    [(class_name, probability * 100) for class_name, probability in probabilities_raw.items()],
                    key=lambda item: item[1],
                    reverse=True,
                )
                image_url = url_for("static", filename=f"uploads/{saved_name}")
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        probabilities=probabilities,
        image_url=image_url,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
