# GPT-2 и RL на Yambda

1 000 пользователей · 4 889 035 прослушиваний · 16 384 трека в словаре · 3 seed

[Отчёт](REPORT.md) · [Примеры выдачи](EXAMPLES.md) · [LaTeX](report/main.tex) · [Воспроизведение](docs/REPRODUCING.md)

## Результаты

| Модель | NDCG@10 | Recall@10 | Completion основной RM | Completion аудитора |
| --- | --- | --- | --- | --- |
| SFT | 0.00688 | 0.01542 | 0.6925 | 0.6780 |
| Continued-CE | 0.00686 | 0.01546 | 0.6923 | 0.6778 |
| Balanced RL | 0.00679 | 0.01505 | 0.6953 | 0.6794 |
| Completion без KL | 0.00157 | 0.00326 | 0.7346 | 0.6940 |

Приведены средние метрик трёх отдельных обучений, completion означает прогноз доли дослушивания.
Первичное сравнение balanced RL с continued-CE даёт Δ NDCG@10 **-0.00008**, 95% ДИ **[-0.00107, +0.00084]**.
Предсказанный completion растёт на **0.00297**, 95% ДИ **[+0.00264, +0.00328]**

Рост предсказанной награды не устанавливает улучшение пользовательского поведения.
Сохранение релевантности в пределах 5% также не подтверждено тестовым интервалом

[Все варианты и seed](artifacts/main/evaluation/test/summary.csv) · [Парные интервалы](artifacts/main/paired_comparisons.json) · [Выбор до test](artifacts/main/selection.lock.json)

## Запуск

Проверенная среда: Python 3.9.6, Torch 2.5.1, macOS arm64, CPU, 4 потока.
Для закреплённых зависимостей используйте Python 3.9-3.11.
Полный запуск требует около 10 ГБ диска и 16-32 ГБ RAM, исходный файл занимает около 426 МБ

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce PYTHON=python
```

Команда выполняет подготовку данных, SFT, обучение RM, 21 дообучение и оценку.
Веса и большие кэши создаются при запуске и исключены из Git.
Поэтапный запуск и выдача обученной модели описаны в [инструкции](docs/REPRODUCING.md)

## Модели и протокол

GPT-2 обучается с нуля на ID треков, контекст содержит до 64 прошлых событий.
Долю дослушивания предсказывают три CatBoost-регрессора, будущую активность отдельный классификатор.
ExtraTrees служит аудитором, обученным на откликах другой половины пользователей.
RL использует on-policy REINFORCE с четырьмя траекториями по пять действий и KL к замороженной SFT

Validation выбирает параметры и чекпойнты, confirmation фиксирует итоговый вариант до test.
Ранжирование проводится по всему словарю, неизвестные цели остаются в общей метрике с нулевым вкладом.
Исходные планы сохранены в [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) и [DIAGNOSTIC_PLAN.md](DIAGNOSTIC_PLAN.md), их контрольные суммы входят в фиксацию выбора

## Проверка

```sh
make check
make tests PYTHON=python
python -m seqrl.validate --final
```

`make check` проверяет документы, контрольные суммы и пересчитывает сохранённые метрики без данных и весов.
Unit-тесты проверяют временные префиксы, return-to-go и вероятности при генерации.
Полный аудит требует восстановленных данных, весов и кэшей.
Результаты выполненного аудита находятся в [validation_final.json](artifacts/main/validation_final.json)

`make bundle` создаёт `submission/recsys_rl.zip` с кодом, компактными результатами и LaTeX-отчётом без данных и весов

## Навигация по коду

| Модуль | Назначение |
| --- | --- |
| [data.py](seqrl/data.py) | Закреплённый источник и полные истории пользователей |
| [build.py](seqrl/build.py) | Словарь, временные префиксы и оценочные запросы |
| [sft.py](seqrl/sft.py) | Обучение GPT-2 и выбор архитектуры |
| [reward.py](seqrl/reward.py) | Основная RM, аудитор и качество на logged actions |
| [rl.py](seqrl/rl.py) | REINFORCE, KL и continued-CE |
| [evaluate.py](seqrl/evaluate.py) | Confirmation, фиксация выбора и test |
| [rollout.py](seqrl/rollout.py) | Пять шагов при разных переходах контекста |
| [session_evaluate.py](seqrl/session_evaluate.py) | Стохастические траектории и дисконтированная награда |

Отчёт в [report/sections](report/sections/), оформление задано в [preamble.tex](report/preamble.tex).
Инструкция по сборке: [report/README.md](report/README.md)
