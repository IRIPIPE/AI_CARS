# Папка для исходного датасета

Поместите сюда распакованный датасет Kaggle. Проект поддерживает два формата:

- папки классов напрямую: `data/raw/ClassName/*.jpg`;
- готовые split-папки: `data/raw/train/ClassName/*.jpg`, `data/raw/validation/ClassName/*.jpg`, `data/raw/test/ClassName/*.jpg`.

Первоначальный источник:

- Vehicle Images Dataset: https://www.kaggle.com/datasets/lyensoetanto/vehicle-images-dataset

Альтернативный источник для классификации типа кузова:

- Car Body Types Images Dataset: https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped

Пример загрузки через Kaggle API:

```bash
kaggle datasets download -d lyensoetanto/vehicle-images-dataset
unzip vehicle-images-dataset.zip -d data/raw/
```

Для альтернативного датасета:

```bash
kaggle datasets download -d ademboukhris/cars-body-type-cropped
unzip cars-body-type-cropped.zip -d data/raw/
```

Не смешивайте несколько датасетов внутри одной папки `data/raw/`. Если нужно сравнить разные наборы данных, распакуйте их в разные директории и передавайте путь через `--data-dir`.
