# Папка для исходного датасета

Поместите сюда распакованный датасет Kaggle `Vehicle Images Dataset`.

Источник: https://www.kaggle.com/datasets/lyensoetanto/vehicle-images-dataset

Пример загрузки через Kaggle API:

```bash
kaggle datasets download -d lyensoetanto/vehicle-images-dataset
unzip vehicle-images-dataset.zip -d data/raw/
```

После распаковки внутри `data/raw/` должны находиться папки классов с изображениями транспортных средств. Скрипты проекта также умеют искать папку с классами на один или несколько уровней глубже, если архив распаковался во вложенную директорию.
