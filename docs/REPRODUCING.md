# Воспроизведение GPT-2 и RL

[README](../README.md) · [Конфигурация](../configs/main.json) · [Источник данных](../data/SOURCE_MANIFEST.json)

## Полный запуск

Команды выполняются из корня репозитория

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce PYTHON=python
```

Проверенная среда: Python 3.9.6, Torch 2.5.1 CPU, 4 потока.
Для закреплённых пакетов используйте Python 3.9-3.11.
Среда сохранена в [environment_sft.json](../artifacts/main/environment_sft.json) и [environment_rl.json](../artifacts/main/environment_rl.json)

При первом `seqrl.build` поставляемая папка `artifacts/main` переносится в `artifacts/reference_main_N`.
Признак такого снимка хранится в `ARCHIVE_RESULTS_ONLY.json`.
Новый запуск создаёт собственные веса, кэши и фиксацию выбора

Время fit исходного опыта: 75.2 мин для SFT с подбором ширины, 0.9 мин для RM и 57.8 мин для 21 дообучения.
Это сумма измеренных обучений, часть запусков выполнялась параллельно, подготовка и оценка учтены не полностью

## Поэтапный запуск

```sh
python -u -m seqrl.data
python -u -m seqrl.build
python -u -m seqrl.sft
python -u -m seqrl.reward
python -u -m seqrl.rl
python -u -m seqrl.evaluate
python -u -m seqrl.rollout
python -u -m seqrl.session_evaluate
python tools/run_generated.py --output EXAMPLES.md -- python -u -m seqrl.examples
make tests PYTHON=python
python -u -m seqrl.validate --final
```

| Этап | Выход |
| --- | --- |
| `build` | Словарь, префиксы и аудит данных |
| `sft`, `reward`, `rl` | Веса, кривые обучения и записи о запусках |
| `evaluate` | Confirmation, `selection.lock.json`, затем test |
| `rollout`, `session_evaluate` | Greedy-диагностика и стохастические траектории |
| `examples` | Пять фиксированных пользователей и частоты top-1 |
| `validate --final` | Данные, контрольные суммы и пересчёт метрик |

Повторный запуск использует кэши при совпадении fingerprints.
После фиксации выбора изменение весов или конфигурации останавливает оценку

Для другого дизайна задайте новый `output_dir` и передавайте `--config configs/new.json` каждому этапу.
У Makefile есть эквивалент `CONFIG=configs/new.json`.
Выбор когорты выполняется отдельно в `seqrl.data`, текущие обработанные данные закреплены за 1 000 пользователями и seed 314

## Продолжение и пример выдачи

```sh
python -u -m seqrl.rl --seeds 42 --only rl_completion_005
python -m seqrl.demo
python -m seqrl.demo --method rl_completion_005 --seed 43
```

Эти команды требуют обученных весов и кэшей.
Финальная оценка использует все заданные варианты и три seed

## Отчёты

Редактируемый LaTeX собирается отдельно

```sh
make latex
```

`make generated-report` формирует машинный Markdown в `output/generated/REPORT.md` и обновляет графики по сохранённым результатам.
Для этого не нужны обученные веса.
Примеры выдачи при полном запуске также записываются в `output/generated/`, отредактированный `EXAMPLES.md` сохраняется.
Папка `report/` не перезаписывается

`make check` проверяет компактные результаты и документы без скачивания данных и обучения.
`make bundle` создаёт `submission/recsys_rl.zip` с кодом, результатами, Markdown и исходниками LaTeX.
Архив не содержит данные, веса, большие кэши и прежние PDF-экспорты

После нового обучения перенесите изменившиеся числа и рисунки из машинного отчёта в редактируемые Markdown и LaTeX перед упаковкой

Проверка контрольных сумм собранного архива:

```sh
python3 tools/package_submission.py --verify submission/recsys_rl.zip
```
