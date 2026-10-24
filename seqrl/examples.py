"""Fixed random qualitative cases and globally frequent top-1 recommendations."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import dump,load_config
from .reward import RewardModels
from .sft import read_npz


def main(config):
    out=Path(config["output_dir"])
    lock=json.loads((out/"selection.lock.json").read_text())
    z=read_npz(out/"cache/states_test.npz")
    _,first=np.unique(z["uid"],return_index=True)
    indices=np.sort(np.random.default_rng(812).choice(first,min(5,len(first)),replace=False))
    rm=RewardModels(config)
    ids=np.r_[[-1,-1,-1],rm.features.catalog["item_ids"]]
    support=read_npz(out/"cache/reward_baselines.npz")["action_count"]
    methods=list(dict.fromkeys(["sft","continued_ce","rl_completion_free","rl_completion_020","rl_balanced_010",lock["selected_method"]]))
    records=[];exposures=[]
    for name in methods:
        top=read_npz(out/"evaluation/test"/f"{name}_42_top20.npz")
        assert np.array_equal(top["qid"],z["qid"])
        tokens=top["tokens"]
        counts=np.bincount(tokens[:,0],minlength=len(ids))
        valid=np.flatnonzero(counts[3:]>0)+3
        most=valid[np.argsort(-counts[valid],kind="stable")[:5]]
        for token in most:
            exposures.append({"method":name,"token":int(token),"item_id":int(ids[token]),"query_share":float(counts[token]/len(tokens)),
                "duration":float(rm.features.duration[token]),"train_listens":int(rm.features.popularity[token]),"optimizer_rm_labels":int(support[token])})
        for i in indices:
            actions=tokens[i,:5]
            p=rm.predict(np.repeat(z["contexts"][i:i+1],5,axis=0),np.repeat(z["seen"][i:i+1],5,axis=0),np.repeat(z["history_count"][i],5),actions,True)
            for rank,a in enumerate(actions):
                records.append({"qid":int(z["qid"][i]),"uid":int(z["uid"][i]),"method":name,"seed":42,"rank":rank+1,"item_id":int(ids[a]),
                    "duration":float(rm.features.duration[a]),"completion":float(p["completion"][rank]),"audit_completion":float(p["audit_completion"][rank]),
                    "seen":bool(z["seen"][i,a]),"known_positive":bool(z["positives"][i,a])})
    target=out/"evaluation/examples";target.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(exposures).to_csv(target/"top1_exposure.csv",index=False)
    pd.DataFrame(records).to_csv(target/"fixed_cases.csv",index=False)
    dump(target/"case_selection.json",{"seed":812,"rule":"5 uniform users among test-query users; first sampled query of each selected user; no metric-dependent selection",
        "qids":z["qid"][indices].tolist(),"uids":z["uid"][indices].tolist(),"model_seed":42})
    events=pd.read_parquet(Path(config["data_dir"])/"events.parquet")
    users={int(u):p for u,p in events.groupby("uid")}
    repeated,total=0,0
    for uid in np.unique(z["uid"]):
        p=users[int(uid)];times=p.timestamp.to_numpy();items=p.item_id.to_numpy();organic=p.is_organic.to_numpy()
        seen_raw=set();stop=0
        for i in np.flatnonzero(z["uid"]==uid):
            cut=z["timestamp"][i];end=int(np.searchsorted(times,cut,side="left"))
            seen_raw.update(items[stop:end].tolist());stop=end
            after=int(np.searchsorted(times,cut,side="right"))
            positives=np.unique(items[end:after][organic[end:after]==0])
            assert len(positives)==z["all_positives"][i]
            repeated+=sum(int(a) in seen_raw for a in positives);total+=len(positives)
    dump(target/"observed_target_repeats.json",{"all_positive_items":total,"all_positive_previously_seen":repeated,
        "all_positive_history_repeat_rate":repeated/total,"known_positive_history_repeat_rate":float((z["positives"]&z["seen"]).sum()/z["positives"].sum()),
        "definition":"Distinct positive raw IDs at each sampled query; seen only before timestamp, including organic/unknown-vocabulary history"})
    text=["# Примеры выдачи\n","Пять пользователей выбраны seed 812 до просмотра их выдач; первый из оценочных запросов каждого пользователя. Здесь только seed 42, это иллюстрации, не дополнительная оценка эффекта. IDs анонимны; названия/жанры по ним не восстанавливались.\n"]
    for i in indices:
        qid=int(z["qid"][i]);ctx=z["contexts"][i];recent=[str(int(ids[t])) if t>=3 else "UNK" for t in ctx[ctx>=2][-8:]]
        text.append(f"## uid {z['uid'][i]}, qid {qid}\n\nПоследние известные токены: {', '.join(recent)}.\n")
        text.append("| Модель | Ранг | Item ID | Длительность | RM | Audit | Был в истории | Logged positive |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in records:
            if r["qid"]==qid:
                text.append(f"| {r['method']} | {r['rank']} | {r['item_id']} | {r['duration']:.0f} | {r['completion']:.3f} | {r['audit_completion']:.3f} | {r['seen']} | {r['known_positive']} |")
        text.append("")
    Path("EXAMPLES.md").write_text("\n".join(text))
    print("Fixed cases and top-1 exposure items saved",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/main.json")
    main(load_config(p.parse_args().config))
