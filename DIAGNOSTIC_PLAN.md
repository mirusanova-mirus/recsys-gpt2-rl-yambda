# Дополнение к плану: heldout stochastic session score

Добавлено после начала SFT/RL и просмотра validation, до confirmation/test model
metrics. Гиперпараметры, checkpoints и первичное сравнение не меняются. Это
дополнительная диагностика, не новый критерий выбора.

Кроме заранее заданного greedy rollout stress test, считаем stochastic G5
на 256 равномерно выбранных test queries (seed 622), без условия на будущую
непрерывность сессии. Четыре траектории на запрос, temperature 1, gamma .9;
общие uniforms seed 623. Две искусственные динамики: append каждого действия
и frozen context. Нормировка return на сумму discount weights для шкалы [0,1]
у completion. Сравниваются optimizer/auditor completion, seconds и balanced
utilities, повторы и duration; отдельно шаги 1 и 5. Фиксированное primary
balanced vs continued-CE сохраняется, интервалы по пользователям условны на
выбранных обучениях и draws. Эти scores не оценивают реальные пользовательские
траектории. Checkpoints выбирались по first-step validation utility; horizon
5 относится к train objective и этой дополнительной финальной диагностике.

Анализ поддержки reward model (число train откликов для item ID) и срезы
калибровки добавлены до первых RL updates. Они не используются для изменения
функций награды или отбора моделей. В приложении сохраняются все варианты.

После первых validation checkpoints seed 42 генерация ускорена KV cache:
математическая политика, префиксы, loss, budget и правила выбора не меняются.
Равенство logits для padded/unpadded histories проверяется unit test и каждого
training rollout — сравнением sampled/replayed log probabilities. Первые
engineering runs перенесены в artifacts/engineering_sampling_v1, в итоговые
таблицы не входят; все основные дообучения заново запускаются одним кодом.
Test model metrics при этом ещё не вычислялись. Это изменение реализации
sampling, а не выбор reward/hyperparameters по результату.

Качественные примеры: пять равномерно выбранных пользователей из test-query
cohort, seed 812, первый из sampled queries каждого пользователя; фиксируется
кодом до вычисления test metrics. Показывается seed 42, top-5 по нескольким
заранее заданным вариантам. Агрегированные выводы строятся по полной оценочной
выборке, а не по этим иллюстрациям.
