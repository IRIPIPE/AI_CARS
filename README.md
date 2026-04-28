# Разработка CNN для классификации типа транспортного средства

Учебный лабораторный проект по разработке, обучению и тестированию сверточной нейронной сети на Python с использованием TensorFlow/Keras.

## Цель проекта

Получить практические навыки проектирования нейронной сети для обработки изображений: подготовить данные, построить CNN-модель, обучить ее, оценить качество классификации и разработать простой пользовательский интерфейс для распознавания типа транспортного средства по изображению.

## Описание задачи

Нейронная сеть получает на вход изображение транспортного средства и определяет один из классов:

- Big Truck
- City Car
- Multi Purpose Vehicle
- Sedan
- Sport Utility Vehicle
- Truck
- Van

Классы определяются автоматически по подпапкам датасета, поэтому фактические имена классов будут соответствовать структуре распакованного архива.

## Источник датасета

Используется открытый датасет Kaggle:

- Название: Vehicle Images Dataset
- Ссылка: https://www.kaggle.com/datasets/lyensoetanto/vehicle-images-dataset
- Kaggle slug: `lyensoetanto/vehicle-images-dataset`

## Структура проекта

```text
vehicle_type_cnn_project/
├── data/
│   └── raw/
│       └── README.md
├── models/
│   └── .gitkeep
├── reports/
│   ├── figures/
│   │   └── .gitkeep
│   └── report_materials.md
├── src/
│   ├── config.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── uploads/
│           └── .gitkeep
├── notebooks/
│   └── experiment.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Установка зависимостей

Рекомендуется использовать виртуальное окружение. Для TensorFlow нужен Python совместимой версии. На момент подготовки проекта пакет TensorFlow на PyPI публикуется для Python 3.10, 3.11, 3.12 и 3.13. Python 3.14 не подходит: при установке будет ошибка `No matching distribution found for tensorflow`.

Проверка версии:

```bash
python --version
```

Если установлен Python 3.14, создайте окружение на Python 3.13 или 3.12. Один из удобных вариантов для Linux/Arch Linux — `pyenv`:

```bash
pyenv install 3.13.5
pyenv local 3.13.5
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если Python 3.12 или 3.13 уже установлен в системе:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Обычная установка в новом окружении:

```bash
cd vehicle_type_cnn_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для Windows:

```bash
cd vehicle_type_cnn_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Скачивание датасета

Главный сценарий: скачать датасет вручную с Kaggle и распаковать его в папку `data/raw/`.

Также можно использовать Kaggle API:

```bash
kaggle datasets download -d lyensoetanto/vehicle-images-dataset
unzip vehicle-images-dataset.zip -d data/raw/
```

Скрипты проекта ожидают, что внутри `data/raw/` или во вложенной папке будут директории классов, например:

```text
data/raw/
├── Big Truck/
├── City Car/
├── Multi Purpose Vehicle/
├── Sedan/
├── Sport Utility Vehicle/
├── Truck/
└── Van/
```

## Проверка и подготовка датасета

Скрипт проверяет наличие изображений, автоматически определяет классы по папкам, выводит количество изображений и показывает разбиение на train/validation/test.

```bash
python src/prepare_dataset.py
```

Если датасет лежит в другой папке:

```bash
python src/prepare_dataset.py --data-dir path/to/dataset
```

## Обучение модели

```bash
python src/train_model.py
```

Дополнительные параметры:

```bash
python src/train_model.py --head-epochs 8 --fine-tune-epochs 20 --fine-tune-layers 40 --batch-size 32 --image-size 224
```

Если нужно мягко компенсировать дисбаланс классов, можно включить веса классов:

```bash
python src/train_model.py --class-weight-mode sqrt
```

Режим `balanced` сильнее влияет на малые классы, но может снижать общую accuracy на несбалансированном датасете.

Во время обучения используются:

- нормализация пикселей в диапазон `[0, 1]`;
- изменение размера изображений до `224x224`;
- аугментация обучающей выборки;
- одна CNN-модель на базе `EfficientNetB0` с предобученными весами ImageNet;
- двухэтапное обучение: сначала классификационная голова, затем fine-tuning верхних слоев;
- опциональные class weights для учета дисбаланса классов;
- optimizer `Adam`;
- loss `sparse_categorical_crossentropy`;
- metric `accuracy`;
- сохранение лучшей модели по `val_accuracy`, `EarlyStopping`, `ReduceLROnPlateau`.

После обучения сохраняются:

- модель: `models/vehicle_type_cnn.keras`;
- словарь классов: `models/class_indices.json`;
- параметры обучения: `models/training_config.json`;
- график accuracy/loss: `reports/figures/training_history.png`.

## Оценка модели

```bash
python src/evaluate_model.py
```

Скрипт загружает сохраненную модель, формирует тестовую выборку тем же способом, считает accuracy, выводит `classification report` с precision, recall и f1-score, а также сохраняет confusion matrix:

```text
reports/figures/confusion_matrix.png
```

## Предсказание по одному изображению

```bash
python src/predict.py --image path/to/image.jpg
```

Скрипт выводит предсказанный класс и вероятности по всем классам.

## Запуск Flask-интерфейса

Перед запуском интерфейса модель должна быть обучена и сохранена в `models/vehicle_type_cnn.keras`.

```bash
python app/app.py
```

Затем открыть в браузере:

```text
http://127.0.0.1:5000/
```

Интерфейс позволяет загрузить изображение, показывает его на странице, выводит предсказанный класс и вероятности по классам в процентах.

## Описание модели

В проекте используется одна сверточная нейронная сеть на базе `EfficientNetB0`. Это CNN-архитектура с предобученными весами ImageNet, к которой добавлена собственная классификационная голова для классов транспортных средств.

- входной слой для изображений `224x224x3`;
- восстановление диапазона пикселей `0..255` перед подачей в EfficientNetB0;
- предобученный сверточный backbone `EfficientNetB0`;
- слой `GlobalAveragePooling2D`;
- `BatchNormalization`;
- `Dropout`;
- полносвязный слой `Dense(256, activation='relu')`;
- выходной слой `Dense(num_classes, activation='softmax')`.

CNN выбрана потому, что изображения имеют пространственную структуру. Сверточные слои выделяют локальные признаки: границы, контуры, формы кузова, колеса, окна и другие детали. Использование EfficientNetB0 позволяет начать обучение не с нуля, а с уже сформированных визуальных признаков, что обычно повышает качество классификации. Softmax подходит для многоклассовой классификации.

Обучение выполняется в два этапа. На первом этапе замороженный EfficientNetB0 используется как извлекатель признаков, и обучается только классификационная голова. На втором этапе размораживаются верхние слои EfficientNetB0 и выполняется fine-tuning с малой скоростью обучения.

## Ожидаемый результат

После выполнения проекта пользователь получает обученную модель, метрики качества на тестовой выборке, графики процесса обучения, confusion matrix и локальное веб-приложение для практического распознавания типа транспортного средства по изображению.
