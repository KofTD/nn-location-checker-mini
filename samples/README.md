# Скрипты
Краткое описание скриптов, для более полной информации смотрите ниже

- [`train_model.py`](#тренировка-модели) — тренировка собираемой модели
- [`run_experiment.py`](#эксперимент) — тренировка модели с записью результатот тренировки в формате csv
- [`show_dataset.py`](#вывод-datasetа) — вывод случайных 25 изобржений dataset'а с подписями

# Тренировка модели

> [!IMPORTANT]
> Все скрипты запускаются из виртуального окружения, поэтому здесь и далее префикс `(.venv)` перед коммандами будет опущен в угоду читаемости

```bash
$ python train_model.py -trd <train_dataset_folder>
                        -ted <test_dataset_folder>
                        -c <config_file>
                        -lf <log_folder>
                        -ln <log_name>
                        -s <size> <size>
```

## Аргументы
- `train_dataset_folder` — папка с изображениями в формате <XX_NameOfASight>, например `01_NizhnyNovgorodKremlin`
- `test_dataset_folder` — папка с изображениями в формате <XX_NameOfASight>, например `01_NizhnyNovgorodKremlin`
- `config_file` — конфигурационный toml файл, пример:
```toml
[macro_parameters]
batch_size = 64
epochs = 2

[model]
name = "AlexNet"
end = 2
classifier = "./classifiers/alexnet_classifier.json"

[optimizer]
name = "SGD"
learning_rate = 0.001

[loss_function]
name = "CrossEntropyLoss"
``` 
- `log_folder` — папка, где вы ожидаете увидеть логи
- `log_name` — имя файла без расширения (stem)
- `size size` — ширина и высота картинок, подаваемых нейросети

## Эксперимент
```bash
$ python train_model.py -trd <train_dataset_folder>
                        -ted <test_dataset_folder>
                        -c <config_file>
                        -lf <log_folder>
                        -ln <log_name>
                        -s <size> <size>
                        -o <output_file>
```

### Аргументы
- `train_dataset_folder` — папка с изображениями в формате <XX_NameOfASight>, например `01_NizhnyNovgorodKremlin`
- `test_dataset_folder` — папка с изображениями в формате <XX_NameOfASight>, например `01_NizhnyNovgorodKremlin`
- `config_file` — конфигурационный toml файл, пример:
```toml
[macro_parameters]
batch_size = 64
epochs = 2

[model]
name = "AlexNet"
end = 2
classifier = "./classifiers/alexnet_classifier.json"

[optimizer]
name = "SGD"
learning_rate = 0.001

[loss_function]
name = "CrossEntropyLoss"
``` 
- `log_folder` — папка, где вы ожидаете увидеть логи
- `log_name` — имя файла без расширения (stem)
- `size size` — ширина и высота картинок, подаваемых нейросети
- `output_file` — csv файл, в который будет записан результат эксперимента

## Вывод dataset'а
```bash
$ python show_dataset.py -d <dataset_folder>
                         -s <size> <size>
```

### Аргументы
- `dataset_folder` — папка с изображениями в формате <XX_NameOfASight>, например `01_NizhnyNovgorodKremlin`
- `size size` — ширина и высота картинок, которые будут отображены
