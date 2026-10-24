# Второе задание: GPT-2, reward model и RL

> Редактируемый отчёт: [LaTeX-проект](report/main.tex), [разделы](report/sections/), [инструкция](report/README.md), [PDF из LaTeX](report/report.pdf). Впишите автора в `report/metadata.tex`. Сборка: `make latex`.


Эксперимент на реальном **Yambda-50M**: 1000 случайных слушателей, все их
4 889 035 listening events, GPT-2 с токенами-ID, модель доли дослушивания и
дообучение REINFORCE. Сравнение включает три seed, continued-CE, разные rewards
и KL coefficients, проверку коллапса и пять шагов генерации.

Полные результаты и ограничения: [REPORT.md](REPORT.md).
Версия для чтения: [PDF](output/pdf/yambda_gpt2_rl_research.pdf).

## Что проверяется

| Требование | Реализация и проверка |
| --- | --- |
| GPT-2 next item prediction | `seqrl/sft.py`, GPT2LMHeadModel с нуля, item vocabulary train-only |
| Reward model и heldout quality | `seqrl/reward.py`, CatBoost completion/return, независимые response labels аудитора, MAE/RMSE/R² и calibration |
| RL fine-tuning | `seqrl/rl.py`, on-policy REINFORCE, leave-one-out baseline, frozen-reference KL |
| Поведение и коллапс | entropy, coverage, exposure Gini, common top-1, cross-user Jaccard, повторы, duration |
| Релевантность | Recall/NDCG/MAP@5/10/20 по всем 16 384 трекам словаря; OOV-цели учитываются в общей метрике |
| Долгосрочные цели | цензурированный proxy активности в (t+24h,t+48h], balanced reward и границы causal выводов |
| Пять шагов | фактические префиксы, append-all, frozen context, модельное принятие; диагностический future anchor |
| Reward/relevance trade-off | три KL coefficients, completion/seconds/balanced, matched continued-CE и bandit |
| Воспроизводимость | pinned source/SHA, config, validation/confirmation/test, selection seal, per-query CSV, unit/data audits |

Предсказанная награда на новых действиях не является наблюдаемым улучшением
удовлетворённости или retention. Источник не содержит истинных logging propensities;
несмещённая off-policy/causal оценка не заявляется. Подробности — в отчёте.

## Запуск

Проверенная среда: Python 3.9.6 на macOS arm64, Torch 2.5.1 CPU, 4 threads.
Для установки закреплённых пакетов рекомендуется Python 3.9–3.11. GPU не требуется.
Нужны интернет для первого скачивания (~426 MB), около 10 GB диска и 16–32 GB RAM.
Большие arrays полного seen-history используются для корректной repeat-диагностики.
Полный набор повторений на CPU может занимать несколько часов; измеренные затраты
каждой модели сохранены в JSON, раздел 9 отчёта показывает суммарные fit times.

```bash
cd recsys-gpt2-rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce PYTHON=python
```

Последовательный порядок этапов (эти команды можно запускать отдельно):

```bash
python -u -m seqrl.data
python -u -m seqrl.build
python -u -m seqrl.sft
python -u -m seqrl.reward
python -u -m seqrl.rl
python -u -m seqrl.evaluate
python -u -m seqrl.rollout
python -u -m seqrl.session_evaluate
python -u -m seqrl.examples
python -m unittest discover -s tests -v
python -u -m seqrl.validate --final
python -m seqrl.report
python -m seqrl.pdf_report
python -m seqrl.package
```

`seqrl.evaluate` сначала считает confirmation, выбирает решение по заранее
заданному правилу и сохраняет SHA всех моделей в `selection.lock.json`. Только
затем оценивается test. При повторном запуске веса/конфигурация должны совпасть
с seal. Validation выбирает SFT architecture/checkpoints и RM; test не используется
в этих решениях. До обучения проводился аудит состава данных, включая holdout
coverage — эксперимент не был полностью слепым к составу holdout ID.

Кешированные SFT/RM/RL fits повторно используются при совпадении fingerprints.
Для другого дизайна скопируйте `configs/main.json`, задайте новый `output_dir`
и передайте `--config configs/new.json` каждому этапу, либо
`make reproduce CONFIG=configs/new.json`. Пользовательскую выборку задаёт отдельная
команда `seqrl.data`; текущий `data/processed` закреплён за seed 314/1000 пользователей.

Пример выдачи обученной модели:

```bash
python -m seqrl.demo
python -m seqrl.demo --method rl_completion_005 --seed 43
```

Для частичного продолжения RL без изменения протокола:

```bash
python -u -m seqrl.rl --seeds 42 --only rl_completion_005
```

Финальная оценка требует все варианты и все три seed. Положительный эффект RL
не является условием успешного исследования; все неудачные варианты сохраняются.

## Файлы

- `configs/main.json` — все budgets, seeds и coefficients.
- `EXPERIMENT_PLAN.md` — протокол и исследовательские вопросы.
- `data/SOURCE_MANIFEST.json` — pinned URL, revision, bytes, SHA-256.
- `seqrl/` — скачивание, префиксы, GPT-2, RM, RL, оценка, графики, PDF и упаковка.
- `tests/` — контрпримеры для leakage и математических ошибок.
- `artifacts/main/audit.json` — данные, границы, support и censoring.
- `artifacts/main/sft_records.json`, `reward_models.json`, `rl_records.json` — реальные fits.
- `artifacts/main/evaluation/{confirmation,test}/summary.json` — все seed и метрики.
- `artifacts/main/evaluation/test/*_mean.csv.gz` — per-query means для paired bootstrap.
- `artifacts/main/paired_comparisons.json` — user-cluster confidence intervals.
- `artifacts/main/evaluation/rollouts/` — все режимы и шаги диагностики.
- `artifacts/main/validation_final.json` — аудит источника, временных префиксов и результатов.
- `output/figures/` — графики, построенные из сохранённых таблиц.
- `EXAMPLES.md` — фиксированная случайная выборка выдач и анонимных ID треков.

Raw data, большие caches и веса создаются запуском. Они не входят в submission
archive; компактные empirical results, fingerprints и отчёт включены.

При первом `seqrl.build` после распаковки готовые `artifacts/main` автоматически
переносятся в `artifacts/reference_main_N`: в архиве есть специальный marker.
Новый run создаётся с нуля, поэтому старые seals/fingerprints не конфликтуют с
новыми весами или платформой. В исходной рабочей папке с полными caches такого
marker нет: повторный запуск использует проверенные caches.

Проверка целостности архива:

```bash
python -m seqrl.package --verify submission/recsys_rl.zip
```

## Статьи

- [Petrov & Macdonald, 2024: Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning](https://arxiv.org/abs/2403.04875).
- [Matveev et al., 2026: Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction](https://arxiv.org/abs/2607.02818).
- [Yambda-5B: A Large-Scale Dataset for Music Recommendation](https://arxiv.org/abs/2505.22238) и [dataset card](https://huggingface.co/datasets/yandex/yambda).

Здесь не воспроизводятся teacher/PPO первой статьи и two-tower/multi-step
off-policy correction второй. Реализована самостоятельная постановка, которая
соответствует заданию и явно отделяет surrogate optimization от реального эффекта.

## Репозиторий и редактируемый отчёт

Этот репозиторий содержит код, конфигурации, измеренные результаты и полный LaTeX-отчёт. Сырые данные, большие кэши и обученные веса не входят в Git: они воспроизводятся командами выше. `report/` можно редактировать и собирать отдельно.

Происхождение исходного снимка зафиксировано в `provenance/origin.json`; манифест в этой папке относится к исходному архиву до добавления LaTeX, а не к изменённым пользователем файлам.

Репозиторий создан локально с веткой `main`. Для публикации создайте пустой репозиторий на GitHub, затем выполните в этой папке:

```bash
git remote add origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
git push -u origin main
```
