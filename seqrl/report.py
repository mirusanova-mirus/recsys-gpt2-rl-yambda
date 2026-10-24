"""Readable Russian report and PDF source specification from sealed measurements."""
import argparse
import json
from pathlib import Path
import statistics

import pandas as pd

from .common import dump,load_config
from .plots import create,LABELS


SHORT={"most_popular":"Популярность","recent_frequency":"Частота в контексте","sft":"SFT GPT-2","continued_ce":"Continued CE",
    "rl_completion_free":"Completion, β=0","rl_completion_005":"Completion, β=.05","rl_completion_020":"Completion, β=.20",
    "rl_seconds_010":"Секунды, β=.10","rl_balanced_010":"Balanced, β=.10","bandit_completion_005":"Bandit, β=.05"}


def main(config):
    out=Path(config["output_dir"])
    load=lambda name:json.loads((out/name).read_text())
    audit,lock,sft,reward,comparisons,env,rl,checks=[load(n) for n in ["audit.json","selection.lock.json","sft_selection.json",
        "reward_models.json","paired_comparisons.json","environment_sft.json","rl_records.json","validation_final.json"]]
    summary=load("evaluation/test/summary.json")
    conf=load("evaluation/confirmation/summary.json")
    rmq=load("reward_quality_test.json")
    data=audit["source_stats"]
    target_repeats=load("evaluation/examples/observed_target_repeats.json")
    m=lambda name,metric:summary[name]["mean"][metric]
    selected=lock["selected_method"]
    primary=[r for r in comparisons if r["primary"]]
    ci=lambda metric:next(r for r in primary if r["metric"]==metric)
    interval=lambda r:f"{r['delta']:+.5f} [{r['ci95'][0]:+.5f}; {r['ci95'][1]:+.5f}]"
    figure_dir=create(config)
    sections=[]
    def section(title,paragraphs,tables=None,figures=None):
        sections.append({"title":title,"paragraphs":paragraphs,"tables":tables or [],"figures":figures or []})
    def table(headers,rows):
        return {"headers":headers,"rows":rows}
    ndcg=ci("ndcg@10");completion=ci("expected_completion");auditor=ci("expected_audit_completion")
    evidence="Рост релевантности подтверждается данным условным интервалом." if ndcg["ci95"][0]>0 else "Интервал не подтверждает рост релевантности."
    if ndcg["ci95"][1]<0:
        evidence="Интервал показывает снижение релевантности в данном протоколе."
    section("RL для последовательных рекомендаций",[
        "Второе задание. Реальный эксперимент на Yambda: GPT-2 по ID треков, предиктор доли дослушивания и on-policy REINFORCE. Все числа ниже получены запуском приложенного кода; положительный эффект RL не предполагался заранее.",
        f"Первичное сравнение: balanced RL против continued-CE. NDCG@10: {m('continued_ce','ndcg@10'):.5f} → {m('rl_balanced_010','ndcg@10'):.5f}. Δ и 95% ДИ: {interval(ndcg)}. {evidence}",
        f"Изменение ожидаемого completion основной RM: {interval(completion)}; аудиторской RM: {interval(auditor)}. Обе величины — предсказания моделей на предлагаемых действиях, а не наблюдаемый пользовательский эффект.",
        f"По заранее заданному правилу confirmation выбран {SHORT[selected]}. Это решение принято до вычисления финальных test metrics. Первичное сравнение от выбора не меняется.",
        f"Масштаб: {data['events']:,} записей, 1000 случайных слушателей, словарь 16 384 трека, три seed обучения. Семь вариантов дообучения, frozen SFT и два простых контроля. CPU, без предобученных текстовых весов.",
        "Отчёт отвечает на пять вопросов: изменение поведения; сохранение релевантности и коллапс; долгосрочные цели; ошибки перехода при пяти шагах; компромисс между reward и точностью. Код и команды находятся в README.md. Дата: 24 октября 2026."
    ])
    section("1. Данные и доступная информация",[
        "Источник — yandex/yambda, sequential/50m/listens.parquet, закреплённая revision dd6f3a19eef5866e346c3270e098baa641a44948. Равномерно выбраны 1000 из 9238 пользователей, имеющих строку listens, seed 314. Дополнительного фильтра по активности нет. Сохранены все предоставленные listening events этих пользователей; другие типы действий не скачивались.",
        f"Медиана истории {data['history_quantiles']['0.5']:g} событий; минимум {data['history_quantiles']['0.0']:g}. Органическая доля {data['organic_fraction']:.1%}. Повторы и {data['duplicate_full_records']:,} точных дубликатов сохранены. Они могут отражать особенности источника; для reward label одинаковые user/time/item объединяются средним.",
        f"Время — секунды от начала наблюдения с округлением до 5 секунд. {data['same_timestamp_records']:,} записей имеют уже встречавшийся user/timestamp. Их порядок неизвестен. Префикс всегда timestamp < t. В прошлых группах используется техническая сортировка item ID. Все цели неоднозначных timestamp-групп исключены из LM loss, но остаются в истории и групповых оценочных целях.",
        "Общая временная граница одинакова для всех пользователей. Три последние недели — validation, confirmation и test. Во время оценки доступна реально наблюдённая история до текущего запроса, включая более ранние события этой недели; веса фиксированы. Это prequential-протокол, а не замороженная история на начало недели.",
        "Словарь — top-16 384 по частоте только в train. Остальные ID сохраняются как UNK; PAD/BOS/UNK запрещены среди рекомендаций. Длительность — train median по треку, предполагается доступным стабильным метаданным. Полные истории сохранены, GPT видит до 64 токенов в базовом запросе.",
        "Vocabulary, embeddings и глобальная популярность — фиксированные параметры/статистики, выученные ко времени train_end. Более ранние train-префиксы преобразуются ими ретроспективно; не утверждается их доступность в исторический момент каждого train row. В validation/test они уже известны до начала периода. RL обучается на прошлых префиксах с каталогом train_end, а не воспроизводит исторические deployment policies; для такой задачи потребовался бы release-time каталог."
    ],[table(["Часть","Начало, включено","Конец, исключён","Запросы","Покрытие целей"],[
        [name,b["start_inclusive"],b["end_exclusive"],b["sampled_queries"],f"{b['in_vocabulary_positive_coverage']:.1%}"] for name,b in audit["splits"].items()])])
    section("2. GPT-2 и честное сравнение",[
        f"GPT2LMHeadModel обучен с нуля: 2 causal decoder layers, 4 attention heads, width {sft['width']} (выбор из 96/128 по validation NDCG@10), tied input/output embeddings. Токен — ID трека. Dropout .1 при SFT; AdamW lr .001, batch 32, до 8 эпох, patience 2, clip gradient norm 1.",
        f"В SFT выбраны 30 000 перекрывающихся окон (stride 32) из train. Всего {audit['sft_unambiguous_target_tokens']:,} допустимых target positions на эпоху с учётом повторов в окнах. Loss — next-token CE только для известных, неорганических и однозначных по времени целей. Органические события и UNK входят в контекст. У коротких историй BOS и левый padding; loss на padding отсутствует.",
        "SFT checkpoint выбирается по validation. Для выбранной архитектуры обучены отдельные seed 42/43/44. Все варианты дообучения внутри seed стартуют с одной и той же сохранённой SFT. Continued-CE получает те же 300 updates и те же начальные префиксы. Его set loss = −log Σ_{i∈P_t} π(i|s_t), где P_t — известные неорганические цели timestamp-группы; при одной цели это обычный CE. Тот же loss используется в balanced auxiliary term. Это контроль дополнительного обучения; число FLOPs, actions и затраты времени не приравниваются.",
        "Оценочный запрос — user/timestamp с неорганическими событиями; положительные объекты — уникальные неорганические item ID этой группы. Ранжирование точное по всем 16 384 доступным трекам, без sampled negatives. Повторы разрешены: в музыке знакомый трек может быть релевантен. Поэтому repeat — диагностика, а не автоматически ошибка.",
        "Оценочные запросы равномерно сэмплируются без возвращения из всех подходящих timestamp-групп: 512/1024/2048 на validation/confirmation/test. Для 12 000 RL/continued-CE train states дополнительно требуется хотя бы одна known positive, чтобы определить CE loss. Во временных holdout такого фильтра нет; полностью OOV-запросы сохраняются. Для всех методов используются одни и те же cached queries.",
        f"В test {audit['splits']['test']['sampled_queries']} запросов у {audit['splits']['test']['sampled_users']} пользователей; {audit['splits']['test']['fully_oov_queries']} запросов полностью вне словаря. Общие Recall/NDCG/MAP@5/10/20 включают OOV-цели с нулевым вкладом. Conditional-метрики делят только на известные цели и исключают полностью OOV-запросы. Это разные estimands.",
        "Query-macro NDCG использует бинарную релевантность, ideal DCG по min(K, число всех целей); MAP — сумму precision на hit-позициях, делённую на min(K, число целей). Для conditional вариантов знаменатели пересчитываются по known targets. Дополнительные контроли: train popularity и .9*частота в текущем контексте + .1*train popularity."
        ,"Позитивом считается любой logged неорганический listen, включая малый completion. Поэтому ranking metrics измеряют matching исторических действий logging policy, а не экспертную или причинную удовлетворённость. Снижение NDCG само по себе не доказывает ухудшение пользовательского опыта; концентрация и другие диагностики помогают понять характер изменения."
    ],figures=[{"path":"output/figures/sft_curves.png","caption":"Обучение и выбор SFT checkpoint; test не используется для остановки."}])
    section("3. Reward model: качество на logged actions",[
        "Основная цель — clip(played_ratio_pct/100, 0, 1). В источнике ratio >100 означает повторное воспроизведение; clipping — выбранная функция полезности, а не исправление ошибочных данных. Входы: ID кандидата и последнего трека, train duration/popularity, число прошлых событий, повторы, статистики последних ID, PCA-16 embeddings и сходство кандидата с профилем.",
        f"Текущий played_ratio, completion и будущее поведение отсутствуют во входах. Embeddings фиксированы из SFT seed 42. Completion — ансамбль трёх CatBoost RMSE, depth {reward['selected_depth']}, максимум 350 trees, lr .06; depth 6/8 и early stopping 40 выбираются только на validation. Дополнительные два члена используют bootstrap weights по пользователям.",
        f"Отклики оптимизируемой RM обучены на {reward['train_rows_optimizer']:,} train rows одной случайной половины пользователей; ExtraTrees-аудитор — на {reward['train_rows_auditor']:,} rows другой половины. Разделены response labels, но обе модели используют общее фиксированное представление из SFT. Аудитор не участвует в RL loss или выборе checkpoint. Это проверка чувствительности к surrogate, а не независимая истина.",
        "Для каждого reward row и completion, и 48-часовой return label должны полностью наблюдаться внутри своего split; учитывается консервативное время завершения текущего трека. Reward test содержит 20 000 реальных logged действий с известными item ID. Это условная предиктивная оценка на logging distribution; на новых действиях после RL калибровка не гарантирована.",
        "Обучение RM условно на наличии записи listens. Она не оценивает вероятность принять рекомендацию при показе: в выбранном файле нет непрослушанных показов. Поэтому expected completion политики предполагает, что отклик на предлагаемый объект сопоставим с откликом на logged listen; эту selection/transition гипотезу офлайн-метрики не проверяют.",
        "Top-decile bias — среднее prediction минус outcome среди верхнего дециля данного предиктора. Для константного baseline это все строки из-за ties. Полные calibration tables, predicted/observed values и результаты return proxy сохранены отдельно."
    ],[table(["Completion model","MAE","RMSE","R²","Top-decile bias"],[
        [r["model"],f"{r['mae']:.4f}",f"{r['rmse']:.4f}",f"{r['r2']:.4f}",f"{r['top_decile_bias']:+.4f}"] for r in rmq["completion"]])],
        [{"path":"output/figures/calibration.png","caption":"Калибровка на реально наблюдавшихся test actions. Она не устанавливает counterfactual calibration."}])
    section("4. RL: функция цели и обновление",[
        "Используется on-policy grouped REINFORCE с baseline из трёх других траекторий того же реального префикса. На update выбираются 16 train states, по 4 траектории, по 5 действий. Полный softmax по словарю, temperature 1, gamma .9, AdamW lr .0001, 300 updates, clip norm 1. Для каждого RL run сэмплируются 96 000 действий.",
        "На шаге t: shaped reward = r_t − β log(π(a_t|s_t)/π_SFT(a_t|s_t)). Return-to-go G_t = Σ_{j≥t} γ^(j−t) shaped_reward_j. Advantage — G_t минус средний return остальных трёх независимых rollouts. Loss = −mean(Σ_t γ^t A_t logπ_t)/Σ_t γ^t. Sampling и gradient replay используют одну текущую политику без dropout; после update rollout больше не используется. При continued-CE dropout также выключен, gradient update остаётся включённым.",
        "При append выбранный ID добавляется в контекст как предположение surrogate среды. В frozen bandit состояние не меняется, а baseline строится по немедленным rewards, поскольку действия не имеют будущих последствий. Это matched по числу actions/updates контроль контекстных переходов, а не ещё одна реалистичная модель пользователя.",
        "Completion сравнивается при β=0/.05/.20. Seconds reward = min(duration,600)*completion/300. Balanced = .7*completion + .2*return_proxy + .1*novelty − .25*recent_repeat − .1*RM_std; novelty = 1 − log(1+train_count)/log(1+max_count). Для balanced добавлен .05*CE по реальным целям. Коэффициенты заданы до RL и не подбирались по test.",
        "Checkpoints 100/200/300: максимум собственной ожидаемой validation utility. Balanced сначала требует ≥95% SFT validation NDCG, иначе берёт лучший NDCG и отмечает провал ограничения. Continued-CE выбирается по validation NDCG. Ожидаемые rewards оцениваются 32 Monte Carlo действиями на запрос с общими uniforms для всех политик; ranking считается точно.",
        "Frozen SFT задаёт KL reference, а RM и embeddings не обновляются во время RL. У всех вариантов идентичный каталог и исходные checkpoints. В конфигурации сохранены seeds, horizons, коэффициенты и правило выбора."
    ],[table(["Вариант","Шаги seed 42/43/44","Validation NDCG","SFT floor"],[
        [SHORT[v['name']]," / ".join(str(rl[f"{v['name']}_{s}"]["selected_update"]) for s in config["model_seeds"]),
        f"{statistics.mean(rl[v['name']+'_'+str(s)]['validation']['ndcg@10'] for s in config['model_seeds']):.5f}",
        f"{sum(rl[v['name']+'_'+str(s)]['accuracy_floor_met'] for s in config['model_seeds'])}/3" if v["objective"]=="balanced" else "не задан"]
        for v in config["variants"]])],
        [{"path":"output/figures/rl_curves.png","caption":"Validation checkpoints: награда и релевантность могут меняться в разных направлениях. Floor относится только к balanced."}])
    section("5. Релевантность и компромисс с наградой",[
        "В таблице — среднее метрик трёх отдельных моделей. Вероятности не ансамблируются. Reward означает ожидаемый completion оптимизируемой RM при sample temperature 1; Audit — то же по аудиторской RM. Эти scores относятся к предлагаемым действиям, а NDCG — к фактическим действиям в логах.",
        f"NDCG mean ± SD по seed: SFT {m('sft','ndcg@10'):.5f} ± {summary['sft']['sd']['ndcg@10']:.5f}; continued-CE {m('continued_ce','ndcg@10'):.5f} ± {summary['continued_ce']['sd']['ndcg@10']:.5f}; balanced {m('rl_balanced_010','ndcg@10'):.5f} ± {summary['rl_balanced_010']['sd']['ndcg@10']:.5f}. Полные отдельные seed и все K доступны в summary.json.",
        f"Простые контроли: popularity NDCG {m('most_popular','ndcg@10'):.5f}, recent-frequency {m('recent_frequency','ndcg@10'):.5f}; SFT {m('sft','ndcg@10'):.5f}. Completion с β=.20 даёт средние NDCG {m('rl_completion_020','ndcg@10'):.5f} и RM {m('rl_completion_020','expected_completion'):.4f}; frozen-bandit — {m('bandit_completion_005','ndcg@10'):.5f} и {m('bandit_completion_005','expected_completion'):.4f}. Эти дополнительные test точки описательны и не меняют решение на confirmation.",
        f"Первичное balanced − continued-CE: NDCG {interval(ndcg)}, reward {interval(completion)}, audit reward {interval(auditor)}. Bootstrap парный по пользователям, 2000 повторов; сначала усредняются per-query метрики трёх matched seed. Все запросы resampled пользователя идут вместе. Интервалы условны на этих трёх обучениях и общих MC draws; не учитывают всю вариативность train samples/seed.",
        f"Confirmation правило: порог NDCG {lock['confirmation_ndcg_floor']:.5f}; прошли {', '.join(SHORT[n] for n in lock['confirmation_eligible']) or 'ни один RL вариант'}. Выбран {SHORT[selected]}, test NDCG {m(selected,'ndcg@10'):.5f}, reward {m(selected,'expected_completion'):.4f}. Описание этого результата не заменяет первичное сравнение.",
        "Эмпирический Pareto frontier вычислен по средним test NDCG и optimizer reward. Это описательный анализ, не новое правило выбора модели. Дополнительные интервалы и сравнения исследовательские, без поправки на множественные проверки; отсутствие значимости не доказывает эквивалентность и порог 95% не является статистическим non-inferiority тестом."
        ,"Reward functions имеют разные шкалы: completion в [0,1], seconds после деления на 300 может достигать 2. Поэтому одинаковый β между разными functions не означает одинаковую силу ограничения относительно reward. Balanced меняет несколько компонентов и добавляет CE одновременно; этот эксперимент не выделяет причинный вклад каждой составляющей по отдельности. Для этого нужен отдельный factorial/ablation протокол."
    ],[table(["Модель","NDCG@10","Recall@10","MAP@10","RM","Audit"],[
        [SHORT[n],f"{m(n,'ndcg@10'):.5f}",f"{m(n,'recall@10'):.5f}",f"{m(n,'map@10'):.5f}",f"{m(n,'expected_completion'):.4f}",f"{m(n,'expected_audit_completion'):.4f}"] for n in summary])],
        [{"path":"output/figures/pareto.png","caption":"Компромисс с surrogate reward; error bars — SD трёх seed, не 95% ДИ."}])
    section("6. Как меняется поведение и есть ли коллапс",[
        f"Без KL: entropy {m('sft','policy_entropy'):.3f} → {m('rl_completion_free','policy_entropy'):.3f} nats; доля самого частого top-1 {m('sft','top1_largest_item_share'):.1%} → {m('rl_completion_free','top1_largest_item_share'):.1%}; вероятность уже знакомого трека {m('sft','prob_history_repeat'):.1%} → {m('rl_completion_free','prob_history_repeat'):.1%}. Эти измерения показывают масштаб изменения распределения, но не объясняют причину сами по себе.",
        f"Ожидаемая длительность трека: SFT {m('sft','expected_track_seconds'):.1f}s, completion β=0 {m('rl_completion_free','expected_track_seconds'):.1f}s, seconds {m('rl_seconds_010','expected_track_seconds'):.1f}s. Длину рекомендуется рассматривать совместно с reward и релевантностью: выбор более длинного объекта сам по себе не означает большего времени реального прослушивания.",
        f"Вероятность выбрать трек без единого собственного train response label основной RM: SFT {m('sft','prob_no_optimizer_rm_item_labels'):.1%}, completion β=0 {m('rl_completion_free','prob_no_optimizer_rm_item_labels'):.1%}, balanced {m('rl_balanced_010','prob_no_optimizer_rm_item_labels'):.1%}. Такие треки известны GPT-2 и имеют метаданные, но RM обобщает на них без item-specific откликов. Среднее |optimizer−auditor| на sampled actions: SFT {m('sft','expected_rm_disagreement'):.4f}, β=0 {m('rl_completion_free','expected_rm_disagreement'):.4f}.",
        f"Среди фактических test targets {target_repeats['all_positive_history_repeat_rate']:.1%} уже встречались в полной предыдущей истории; среди known targets — {target_repeats['known_positive_history_repeat_rate']:.1%}. Это свойство музыкальных логов объясняет, почему штраф за повторы может конфликтовать с matching-next-item NDCG. Оно не доказывает, что повторы полезны каждому пользователю или что разнообразие ухудшает удовлетворённость.",
        "Coverage — доля словаря, появившаяся в top-10 хотя бы одного запроса. Exposure Gini — неравномерность top-10 показов по словарю. Top1 share — доля запросов с самым частым первым объектом. Cross-user Jaccard — сходство top-10 разных пользователей (первый из отобранных запросов пользователя; фиксированные случайные пары). Высокая concentration совместно со сходством выдач диагностирует обезличивание.",
        "RM_std измеряет расхождение bootstrap-моделей, а optimizer/auditor gap — чувствительность к другому предиктору. Ни один показатель по отдельности не доказывает reward hacking. Для такого вывода нужны реальные outcomes на новых действиях либо рандомизированная онлайн-проверка. Контроль train-item-mean помогает увидеть, насколько RM превосходит простую популярностную оценку отклика."
        ,"В этом запуске вариант без KL смещается к более коротким трекам и общей выдаче для разных пользователей. Это частичный глобальный коллапс; вероятность повторить знакомый пользователю трек при этом снижается. Повтор знакомого объекта, повтор внутри сгенерированного списка и один общий объект у разных пользователей — разные явления. Balanced штрафует повторы только в последних 64 токенах, тогда как таблица P(repeat) использует всю прошлую историю."
    ],[table(["Модель","Покрытие","Gini","Top1 share","Jaccard","Repeat P"],[
        [SHORT[n],f"{m(n,'catalog_coverage@10'):.1%}",f"{m(n,'exposure_gini@10'):.3f}",f"{m(n,'top1_largest_item_share'):.1%}",f"{m(n,'cross_user_jaccard@10'):.3f}",f"{m(n,'prob_history_repeat'):.1%}"] for n in summary])],
        [{"path":"output/figures/behavior.png","caption":"Разные признаки коллапса измеряют разные свойства политики."}])
    roll=pd.read_csv(out/"evaluation/rollouts/summary.csv")
    rsel=roll[(roll.method==selected)&(roll.step==5)]
    greedy_selected={r["mode"]:r for r in rsel.to_dict("records")}
    section("7. Пять шагов: рекомендация не равна действию",[
        "Диагностика использует 256 test-префиксов с пятью следующими одиночными неорганическими событиями и разрывами ≤30 минут. Это специальный срез непрерывных рекомендационных последовательностей; результаты не обобщаются на все сессии. Сравниваются: фактический префикс на каждом шаге; слепое добавление каждого top-1; неизменный префикс; принятие с вероятностью predicted completion.",
        "Все четыре режима начинают с одного реального состояния. Политика в диагностике выбирает top-1, тогда как RL использовал stochastic sampling. В model-acceptance режиме общие uniforms; отклонённый ID не попадает в историю. Это эвристика перехода, а не обученная и проверенная модель принятия рекомендации: дослушивание и выбор объекта — разные события.",
        "NDCG_anchor сравнивает выдачу с реально следующим logged item при разных искусственных контекстах. После первого шага это stress test согласованности с фактической траекторией, а не unbiased counterfactual relevance. Реальный пользователь при новой рекомендации мог выбрать другое действие или уйти; его последующие фактические действия нельзя считать outcomes новой политики.",
        "KL к teacher-forced контексту измеряет накопившийся сдвиг распределения рекомендаций, same-top1 — долю совпадений первого объекта, within-list repeat — повтор среди уже предложенных объектов. RM на синтетических историях может становиться чрезмерно уверенной даже при ухудшении anchor relevance. Для реалистичного session return нужны отклик, organic transition, пропуски и stop/return model, а затем онлайн-валидация."
        ,f"Наблюдаемый шаг 5 выбранной модели: append anchor NDCG {greedy_selected['append_all']['ndcg@10_anchor']:.5f} против {greedy_selected['teacher_forced']['ndcg@10_anchor']:.5f} при actual prefix; KL={greedy_selected['append_all']['kl_to_teacher_context']:.3f}. Greedy RM completion {greedy_selected['append_all']['optimizer_greedy_completion']:.4f} против actual-prefix {greedy_selected['teacher_forced']['optimizer_greedy_completion']:.4f}. Здесь в greedy режиме синтетический переход снижает и anchor matching, и RM score; рост награды на каждом шаге не наблюдается."
    ],[table(["Выбранная модель, шаг 5","Anchor NDCG","RM top1","Audit top1","KL к actual"],[
        [r["mode"],f"{r['ndcg@10_anchor']:.5f}",f"{r['optimizer_greedy_completion']:.4f}",f"{r['auditor_greedy_completion']:.4f}",f"{r['kl_to_teacher_context']:.4f}"] for r in rsel.to_dict("records")])],
        [{"path":"output/figures/rollouts.png","caption":"Расхождение actual и synthetic префиксов. После шага 1 метрика по future anchor имеет только диагностический смысл."}])
    session=pd.read_csv(out/"evaluation/stochastic_sessions/summary.csv")
    session_ci=load("evaluation/stochastic_sessions/paired_comparisons.json")
    session_primary=next(r for r in session_ci if r["mode"]=="append" and r["metric"]=="discounted_mean_completion")
    section("7а. Ожидаемая награда пяти stochastic шагов",[
        "Отдельная дополнительная диагностика соответствует sampling temperature 1 при обучении. Начальные 256 запросов случайно выбраны из общего test sample без отбора по будущему продолжению сессии. На запрос и seed — четыре траектории, пять шагов, gamma .9. В таблице нормированный return Σγ^t r_t / Σγ^t; это шкала среднего completion, а не суммарные реальные минуты.",
        f"Для append-среды первичная разность balanced − continued-CE: {interval(session_primary)}. Интервал по пользователям условен на этих моделях и четырёх общих MC draws. Его нельзя переносить на реальное поведение пользователей.",
        "Под одним и тем же фиксированным набором весов проверяются append и frozen переходы. Различия показывают чувствительность surrogate return к модели контекстного перехода. Это не причинное сравнение пользовательских сценариев; обе динамики искусственные. Predicted return proxy на пяти шагах относится к перекрывающимся будущим окнам, поэтому его сумму нельзя интерпретировать как пять независимых возвращений.",
        "Эта диагностическая оценка была добавлена после начала validation запусков, до test metrics; выбор моделей не использует её результаты. Основной EXPERIMENT_PLAN.md сохранён, дополнение записано в DIAGNOSTIC_PLAN.md. Checkpoints выбирались по first-step validation utility, хотя training return имеет horizon 5 — дополнительное ограничение согласования train/selection objectives."
    ],[table(["Модель","Переход","G5 RM","G5 Audit","Repeat в списке"],[
        [SHORT[r['method']],r['mode'],f"{r['discounted_mean_completion']:.4f}",f"{r['discounted_mean_audit_completion']:.4f}",f"{r['discounted_mean_within_list_repeat']:.1%}"]
        for r in session.to_dict("records") if r["method"] in list(dict.fromkeys(["sft","continued_ce","rl_completion_free","rl_balanced_010",selected]))])])
    section("8. Долгосрочные цели: что доступно офлайн",[
        "Return proxy y=1, если после текущего действия наблюдалось любое listening event в интервале (t+24h,t+48h]. Окно не пересекает границу split. Вторая CatBoost-голова предсказывает этот label по прошлой истории и кандидату; balanced reward даёт ей вес .2. Это наблюдаемая будущая активность, а не вероятность причинного удержания от выбранного трека.",
        "Для настоящей долгосрочной цели можно задавать discounted return: дослушивания + разнообразие/новизна + индикатор возвращения − риск ухода, либо обучать отдельную survival/stop модель с корректным censoring. Нужны активные дни, длина/завершение сессии, organic activity, skips и отрицательная обратная связь; желательно также экспозиции и propensities. В этой работе скачаны только listens, поэтому skips/likes не восстановлены.",
        "Важное ограничение Yambda: исходный cohort выбран по активности в первых десяти месяцах и наличию действия в последнем. Наша равномерная подвыборка не устраняет этот survivorship bias. Высокая доля return labels не описывает churn всей аудитории. Окно 24–48h не означает отдельный вход в приложение и не измеряет удовлетворённость.",
        "Офлайн можно проверить предиктивное качество proxy, временную устойчивость, поведение политики, чувствительность к reward и коллапс. Нельзя заключить, что изменение предсказанного return увеличило фактический retention. Нет истинных logging propensities или рандомизации: IS/DR, unbiased OPE и causal effect здесь не заявляются.",
        "Онлайн-проверка: user-level randomized A/B, заранее зафиксированные eligible cohort и горизонты, primary D7/D14 return/active days, duration/completion как вторичные, guardrails на skips, однообразие и жалобы. Размер и длительность определяются по baseline rates и минимальному эффекту до запуска; в этой работе A/B не проводился."
        ,f"В первичном сравнении Δ expected return-proxy = {interval(ci('expected_return_proxy'))}. Интервал включает ноль: даже рост этого модельного proxy здесь не подтверждён. Это более узкий вывод, чем утверждение об отсутствии любого влияния на реальное удержание."
    ],[table(["Return model","ROC-AUC","PR-AUC","Brier","Logloss","Observed rate"],[
        [r["model"],f"{r['roc_auc']:.4f}",f"{r['average_precision']:.4f}",f"{r['brier']:.4f}",f"{r['logloss']:.4f}",f"{r['observed_rate']:.1%}"] for r in rmq["return_proxy"]])])
    section("9. Стоимость, надёжность и воспроизведение",[
        f"Среда: Python {env['python'].split()[0]}, Torch {env['packages']['torch']}, Transformers {env['packages']['transformers']}, CatBoost {env['packages']['catboost']}; CPU, 4 threads на процесс. SFT grid и три итоговых seed: {env['fit_seconds']/60:.1f} минут чистого fit. Reward model fits: {reward['fit_seconds']/60:.1f} минут; суммарный fit дообучений {sum(r['fit_seconds'] for r in rl.values())/60:.1f} минут. Это сумма измеренных fit times, без скачивания/подготовки/всей оценки и не оценка FLOPs.",
        "Для RL отдельно записаны 96 000 sampled actions/run, optimizer updates, время и число CE positive targets. Для continued-CE число labels меньше, чем число RL actions: одинаковое число обновлений не даёт равный computational budget. Отдельный baseline замороженной SFT позволяет оценить, добавило ли обычное дообучение пользу.",
        "Часть fits выполнялась параллельно и конкурировала за CPU; это не изолированный hardware benchmark. Для RL seed 43/44 отключён idle busy-wait OpenMP (OMP_WAIT_POLICY=PASSIVE, KMP_BLOCKTIME=0), при тех же четырёх вычислительных threads; runtime_scheduling.json фиксирует различие. Суммарные fit times исключают прерванные инженерные прогоны и не равны всему времени разработки.",
        f"Аудит сравнил все {checks['source_complete_events']:,} записей 1000 выбранных пользователей с исходным parquet, восстановил все {checks['all_lm_windows_reconstructed']:,} LM окон и проверил masking. Для всех cached states проверено history_max < timestamp; по 256 примеров каждого split/kind точно восстановлены prefix, seen mask и labels. SHA источника и моделей проверены; saved test means пересчитаны из per-query CSV.",
        "Восемь unit tests проверяют timestamp ties, causal replay и KV cache, masked labels, discount/leave-one-out baseline и направление градиента, OOV denominator, RM feature fractions и reward objectives. Tests не заменяют data audit. Упакованный архив содержит код, конфигурацию, планы, отчёт и компактные результаты; raw data, большие cache и обученные веса исключены.",
        "Полное воспроизведение: создать Python 3.9+ environment, установить requirements.txt, выполнить make reproduce. Публичный parquet ~426 MB скачивается по закреплённой revision и проверяется SHA. Рекомендуется около 10 GB свободного диска и 16–32 GB RAM; этот запуск выполнен на машине с 32 GB. Точное время зависит от CPU и библиотек. Команда demo печатает top-10 реального heldout prefix после обучения.",
        "Cache reuse проверяет config, source и model hashes; при изменении дизайна нужно новое output_dir. Текущие JSON содержат относительные пути внутри проекта. Reproduction сохраняет новый selection seal до test. Публикация или отправка работы не выполнялась."
    ])
    section("10. Связь со статьями и ограничения",[
        "Petrov & Macdonald (2024), Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning: sequential Next-K и оптимизация свойств списка через RL; teacher-student pretraining и PPO. Здесь реализовано требуемое заданием next-item обучение GPT-2 и REINFORCE, поэтому это самостоятельный небольшой эксперимент по близкой идее, не буквальная репликация GPTRec. Источник: https://arxiv.org/abs/2403.04875",
        "Matveev et al. (2026), Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction: session rewards, feedback model и многошаговая коррекция off-policy обучения retrieval. Здесь нет их two-tower архитектуры, logged propensities или multi-step IS; actions сэмплируются on-policy в фиксированной surrogate среде. Авторы также различают model-based scores и реальные результаты. Источник: https://arxiv.org/abs/2607.02818",
        "Описание Yambda и формулы времени: https://arxiv.org/abs/2505.22238 ; dataset card: https://huggingface.co/datasets/yandex/yambda . Pinned file URL и SHA находятся в data/SOURCE_MANIFEST.json. Код использует поля listens, is_organic, played_ratio_pct и track_length_seconds.",
        "Ограничения: небольшая GPT-2 и ограниченные CPU budgets; 1000 пользователей и зависимые повторные события; словарь покрывает около половины test targets; arbitrary order внутри уже прошлых tied blocks; метаданные трактуются как стабильные; response модели оцениваются на логах другой политики; surrogate transitions и acceptance не валидированы; return proxy условен на активном исходном cohort; три seed и user-bootstrap не покрывают всю неопределённость.",
        "Дизайн не был полностью слепым к составу holdout: до первого обучения словарь 2048 заменён на 16 384 после аудита train coverage/CPU benchmark; проверялось и покрытие heldout target IDs. Model metrics при этом не вычислялись. Пилот 2048 сохранился как data-only design audit и не используется для выбора лучшего результата. Все последующие model choices заданы в EXPERIMENT_PLAN.md и selection.lock.json.",
        "Практический вывод следует из совместного рассмотрения релевантности, концентрации, двух reward estimators и устойчивости контекста. Рост surrogate reward достаточен для подтверждения работы оптимизатора, но недостаточен для заявления об улучшении пользовательского опыта. Следующий этап — более широкий item vocabulary, независимый временной/пользовательский cohort, richer feedback и онлайн-проверка."
    ])
    section("11. Основные выводы",[
        f"Оптимизатор способен менять поведение: completion без KL повышает ожидаемую RM награду с {m('sft','expected_completion'):.4f} до {m('rl_completion_free','expected_completion'):.4f}, но NDCG меняется с {m('sft','ndcg@10'):.5f} до {m('rl_completion_free','ndcg@10'):.5f}. Средняя длительность сокращается до {m('rl_completion_free','expected_track_seconds'):.1f}s, самый частый top-1 занимает {m('rl_completion_free','top1_largest_item_share'):.1%} запросов. Это полезный отрицательный результат: модельная награда и matching наблюдавшихся действий расходятся.",
        f"Balanced выбран до test и даёт небольшой прирост по основной и аудиторской RM. Первичный Δ NDCG {interval(ndcg)}: точечная оценка близка к CE, но это не доказательство эквивалентности. Потеря 5% от CE соответствует {-.05*m('continued_ce','ndcg@10'):.5f}; интервал Δ шире этого масштаба. Для убедительной проверки допустимой потери нужны больше независимых пользователей/периодов и заранее рассчитанная мощность.",
        "Более сильный KL сохраняет персонализацию ценой меньшего reward gain; seconds сдвигает длину объектов, а balanced меняет несколько целей одновременно. Frozen-bandit остаётся конкурентным описательным контролем: само добавление пяти сгенерированных ID не гарантирует пользы многошагового RL.",
        "Долгосрочный proxy предсказуем лишь умеренно и не дал подтверждённого роста в первичном сравнении. Сдвиг пятишагового контекста заметен и зависит от decoding/transition assumptions. Следующий содержательный шаг — проверенная модель принятия/ухода, более широкий каталог и независимый cohort; вывод о реальном пользовательском эффекте требует онлайн-эксперимента."
    ])
    text=["# RL для последовательных рекомендаций\n"]
    for sec in sections:
        text.append("## "+sec["title"]+"\n")
        text.extend(p+"\n" for p in sec["paragraphs"])
        for tab in sec["tables"]:
            text.append("| "+" | ".join(map(str,tab["headers"]))+" |\n| "+" | ".join(["---"]*len(tab["headers"]))+" |")
            text.extend("| "+" | ".join(map(str,row))+" |" for row in tab["rows"])
            text.append("")
        for fig in sec["figures"]:
            text.append(f"![{fig['caption']}]({fig['path']})\n")
    text.append("## Дополнительные таблицы\n")
    text.append("По каждому seed, все K и conditional metrics: artifacts/main/evaluation/test/summary.csv и summary.json. Парные интервалы: paired_comparisons.json. Все validation checkpoints и training diagnostics: curve_*.json. Пять шагов и все modes: evaluation/rollouts/summary.csv.\n")
    text.append("| Модель | NDCG@10 mean ± SD | Conditional NDCG@10 | KL к SFT |\n| --- | --- | --- | --- |")
    for n in summary:
        kl=summary[n]["mean"].get("kl_to_sft")
        text.append(f"| {SHORT[n]} | {m(n,'ndcg@10'):.5f} ± {summary[n]['sd']['ndcg@10']:.5f} | {m(n,'conditional_ndcg@10'):.5f} | {kl:.4f} |" if kl is not None else f"| {SHORT[n]} | {m(n,'ndcg@10'):.5f} ± 0 | {m(n,'conditional_ndcg@10'):.5f} | — |")
    Path("REPORT.md").write_text("\n".join(text))
    dump("output/report_sections.json",sections)
    print("REPORT.md, figures and PDF source specification generated",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/main.json")
    main(load_config(p.parse_args().config))
