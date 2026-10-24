"""Confirmation-only decision, immutable seal, then a single final test comparison."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .assessment import assess
from .common import dump,load_config,seed_all,sha
from .metrics import paired_bootstrap
from .model import load_model,predict
from .reward import RewardModels,quality
from .sft import read_npz


def mean_queries(frames):
    keys=["qid","uid","timestamp","all_positives","known_positives"]
    for frame in frames[1:]:
        assert frame[keys].equals(frames[0][keys])
    result=frames[0][keys].copy()
    for c in frames[0]:
        if c not in keys:
            result[c]=np.mean(np.stack([f[c].to_numpy() for f in frames]),axis=0)
    return result


def method_models(config):
    out=Path(config["output_dir"])
    sft=json.loads((out/"sft_selection.json").read_text())
    rl=json.loads((out/"rl_records.json").read_text())
    result={"sft":{int(s):p for s,p in sft["models"].items()}}
    for v in config["variants"]:
        name=v["name"]
        result[name]={seed:rl[f"{name}_{seed}"]["path"] for seed in config["model_seeds"]}
    return result


def control_scores(name,states,catalog):
    pop=catalog["popularity"].astype(np.float64)
    pop/=pop.sum()
    if name=="most_popular":
        p=np.tile(pop,(len(states["contexts"]),1))
    elif name=="recent_frequency":
        p=np.zeros((len(states["contexts"]),len(pop)),np.float64)
        for i,ctx in enumerate(states["contexts"]):
            counts=np.bincount(ctx[ctx>=3],minlength=len(pop))
            p[i]=.9*counts/max(counts.sum(),1)+.1*pop
            p[i]/=p[i].sum()
    else:
        raise ValueError(name)
    scores=np.log(np.maximum(p,1e-30)).astype(np.float32)
    scores[:,:3]=-1e9
    return scores


def evaluate_split(config,split,methods,reward):
    out=Path(config["output_dir"])
    target=out/"evaluation"/split
    target.mkdir(parents=True,exist_ok=True)
    states=read_npz(out/"cache"/f"states_{split}.npz")
    result={}
    seeds=config["model_seeds"]
    # At most one reference-score matrix per seed is kept in RAM.
    for name in ["most_popular","recent_frequency"]+list(methods):
        frames,records=[],[]
        for seed in (seeds if name in methods else [0]):
            reference=None
            if name in methods:
                model,_=load_model(methods[name][seed])
                scores=predict(model,states["contexts"])
                del model
                if name!="sft":
                    ref,_=load_model(methods["sft"][seed])
                    reference=predict(ref,states["contexts"])
                    del ref
                else:
                    reference=scores
            else:
                scores=control_scores(name,states,reward.features.catalog)
            objective=next((v["objective"] for v in config["variants"] if v["name"]==name),"completion")
            if objective=="ce":
                objective="completion"
            metric,q,top=assess(config,states,scores,reward,objective,reference,audit=split=="test")
            q.to_csv(target/f"{name}_{seed}.csv.gz",index=False)
            np.savez_compressed(target/f"{name}_{seed}_top20.npz",tokens=top,qid=states["qid"])
            frames.append(q);records.append({"seed":seed,**metric})
            print(f"{split} {name}/{seed}: NDCG={metric['ndcg@10']:.5f} RM={metric['expected_completion']:.5f} entropy={metric['policy_entropy']:.3f}",flush=True)
            del scores,reference
            gc.collect()
        qmean=mean_queries(frames)
        qmean.to_csv(target/f"{name}_mean.csv.gz",index=False)
        cols=[c for c in records[0] if c!="seed"]
        result[name]={"mean":{c:float(np.mean([m[c] for m in records])) for c in cols},
            "sd":{c:float(np.std([m[c] for m in records],ddof=1)) if len(records)>1 else 0. for c in cols},"seeds":records}
    dump(target/"summary.json",result)
    pd.DataFrame([{ "method":n,**r["mean"]} for n,r in result.items()]).to_csv(target/"summary.csv",index=False)
    return result


def model_fingerprint(config,methods):
    out=Path(config["output_dir"])
    files={p for models in methods.values() for p in models.values()}
    files|={str(p) for p in (out/"models").glob("reward_*")}
    files|={str(out/"cache/reward_embedding.npy"),str(out/"vocabulary.json")}
    files|={str(out/"cache"/name) for name in ["catalog.npz","reward_baselines.npz","states_confirmation.npz","states_test.npz","reward_test.npz","rollouts_test.npz"]}
    return {p:sha(p) for p in sorted(files)}


def main(config):
    seed_all(config["seed"],config["threads"])
    out=Path(config["output_dir"])
    methods=method_models(config)
    fingerprint=model_fingerprint(config,methods)
    rm=RewardModels(config)
    lock_path=out/"selection.lock.json"
    if lock_path.exists():
        lock=json.loads(lock_path.read_text())
        assert lock["config"]==config and lock["model_sha256"]==fingerprint,"Sealed choices changed"
        confirmation=json.loads((out/"evaluation/confirmation/summary.json").read_text())
    else:
        confirmation=evaluate_split(config,"confirmation",methods,rm)
        floor=config["balanced_accuracy_floor"]*confirmation["continued_ce"]["mean"]["ndcg@10"]
        eligible=[n for n in methods if n not in ["sft","continued_ce"] and confirmation[n]["mean"]["ndcg@10"]>=floor]
        choice=max(eligible,key=lambda n:confirmation[n]["mean"]["expected_completion"]) if eligible else "continued_ce"
        lock={"config":config,"model_sha256":fingerprint,"selected_method":choice,"confirmation_eligible":eligible,
            "confirmation_ndcg_floor":floor,"primary_comparison":config["primary_comparison"],
            "confirmation_summary_sha256":sha(out/"evaluation/confirmation/summary.json"),
            "sealed_unix_time":time.time(),"rule":"RL with greatest optimizer-RM expected completion among models at >=95% continued-CE confirmation NDCG; else continued-CE",
            "test_metrics_computed_before_seal":False}
        lock["protocol_plan_sha256"]=sha("EXPERIMENT_PLAN.md")
        lock["diagnostic_plan_sha256"]=sha("DIAGNOSTIC_PLAN.md")
        lock["evaluation_source_sha256"]={str(p):sha(p) for p in [Path("seqrl/evaluate.py"),Path("seqrl/assessment.py"),Path("seqrl/metrics.py"),Path("seqrl/reward.py"),Path("seqrl/model.py"),Path("seqrl/rollout.py"),Path("seqrl/session_evaluate.py"),Path("seqrl/examples.py")]}
        dump(lock_path,lock)
        print(f"SEALED before test: confirmation choice={choice}; eligible={eligible}",flush=True)
    quality(config,"test")
    test=evaluate_split(config,"test",methods,rm)
    comparisons=[]
    pairs=[tuple(config["primary_comparison"]),("sft","continued_ce"),("sft",lock["selected_method"]),
           ("continued_ce",lock["selected_method"]),("sft","rl_completion_free"),("sft","rl_completion_005"),
           ("rl_completion_005","bandit_completion_005")]
    for a,b in dict.fromkeys(pairs):
        qa=pd.read_csv(out/"evaluation/test"/f"{a}_mean.csv.gz")
        qb=pd.read_csv(out/"evaluation/test"/f"{b}_mean.csv.gz")
        for metric in ["ndcg@10","recall@10","map@10","expected_completion","expected_audit_completion","expected_return_proxy","prob_history_repeat"]:
            ci=paired_bootstrap(qa,qb,metric,config["bootstrap_reps"])
            comparisons.append({"baseline":a,"candidate":b,"primary":(a,b)==tuple(config["primary_comparison"]),**ci})
    dump(out/"paired_comparisons.json",comparisons)
    pareto=[]
    names=list(methods)
    for name in names:
        a=test[name]["mean"]
        dominated=any(test[b]["mean"]["ndcg@10"]>=a["ndcg@10"] and test[b]["mean"]["expected_completion"]>=a["expected_completion"] and
            (test[b]["mean"]["ndcg@10"]>a["ndcg@10"] or test[b]["mean"]["expected_completion"]>a["expected_completion"]) for b in names if b!=name)
        pareto.append({"method":name,"empirical_pareto":not dominated})
    dump(out/"pareto.json",pareto)
    print("Final test tables and paired user bootstrap saved",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/main.json")
    main(load_config(parser.parse_args().config))
