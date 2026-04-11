# nn-location-checker-mini

Набор Python скриптов по fine-tuning исследованию существующих архитектур нейросетей с произвольными классификаторами для определения архитектурных достопримечательностей Нижнего Новгорода.

## Для начала работы

### Требования

python >= 3.12

### Установка и конфигурация

```bash
$ git clone https://github.com/itlab-vision/nn-location-checker-mini.git
$ cd nn-location-checker-mini
$ python -m venv .venv
$ source ./.venv/bin/activate
(.venv) $ pip install .
```

### Структура проекта

- `classifiers` — файлы классификаторов в json формате.
- `samples` — скрипты для обучения, экспериментов и визуализации.
- `training_config.toml` — пример файла конфигурации по проведению тренировки или эксперимента.

### Использоавание

Вам необходимы скрипты, расположенные в дирректории `samples/`.

Для более полной информации прочитайте [`samples/README.md`](samples/README.md).

### Тестирование

```bash
(.venv) $ pip install --group dev
(.venv) $ python -m pytest
```
