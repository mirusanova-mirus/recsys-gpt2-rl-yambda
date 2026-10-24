"""Heldout stochastic five-step returns under two explicit surrogate transitions.

Complements greedy future-anchor diagnostics; does not use future targets to
select the initial state sample and does not claim real user trajectories.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from .common import dump,load_config,seed_all,sha
from .evaluate import method_models,mean_queries
from .metrics import paired_bootstrap
from .model import load_model,predict,CachedPolicy
from .reward import RewardModels,utility
from .sft import read_npz


def main(config):
    out=Path(config["output_dir"])
    lock=json.loads((out/"selection.lock.json").read_text())
    seed_all(990,config["threads"])
    all_states=read_npz(out/"cache/states_test.npz")
    n=min(config["rollout_queries"],len(all_states["contexts"]))
    indices=np.sort(np.random.default_rng(622).choice(len(all_states["contexts"]),n,replace=False))
    states={k:v[indices] for k,v in all_states.items()}
    g,h=config["rl_group"],config["rl_horizon"]
    uniforms=np.random.default_rng(623).random((n*g,h))
    discount=config["rl_gamma"]**np.arange(h)
    rm=RewardModels(config)
    records=[];frames={}
    target=out/"evaluation/stochastic_sessions";target.mkdir(parents=True,exist_ok=True)
    for name,models in method_models(config).items():
        for seed,path in models.items():
            assert sha(path)==lock["model_sha256"][path]
            model,_=load_model(path)
            for mode in ["append","frozen"]:
                ctx=np.repeat(states["contexts"],g,axis=0)
                seen=np.repeat(states["seen"],g,axis=0).copy()
                counts=np.repeat(states["history_count"],g)
                values={k:[] for k in ["completion","audit_completion","return_proxy","seconds","balanced","history_repeat","within_list_repeat","duration"]}
                generated=[]
                caches=[CachedPolicy(model,torch.as_tensor(ctx[i:i+128],dtype=torch.long)) for i in range(0,len(ctx),128)] if mode=="append" else None
                frozen_scores=predict(model,ctx) if mode=="frozen" else None
                previous=None
                for step in range(h):
                    if mode=="append":
                        model.eval()
                        scores=np.concatenate([cached.logits(None if previous is None else torch.as_tensor(previous[i*128:(i+1)*128],dtype=torch.long)).numpy() for i,cached in enumerate(caches)])
                    else:
                        scores=frozen_scores
                    p=softmax(scores,axis=1).astype(np.float64)
                    cdf=np.cumsum(p,axis=1);cdf/=cdf[:,-1:]
                    actions=np.array([np.searchsorted(cdf[i],uniforms[i,step],side="right") for i in range(n*g)],np.int32)
                    actions=np.minimum(actions,p.shape[1]-1)
                    assert (actions>=3).all()
                    recent=ctx[:,-config["context"]:]
                    pred=rm.predict(recent,seen,counts,actions,True)
                    values["completion"].append(pred["completion"])
                    values["audit_completion"].append(pred["audit_completion"])
                    values["return_proxy"].append(pred["return_proxy"])
                    values["seconds"].append(utility(pred,actions,recent,rm.features.catalog,"seconds"))
                    values["balanced"].append(utility(pred,actions,recent,rm.features.catalog,"balanced"))
                    values["history_repeat"].append(seen[np.arange(n*g),actions])
                    values["within_list_repeat"].append(np.zeros(n*g,bool) if not generated else (np.column_stack(generated)==actions[:,None]).any(1))
                    values["duration"].append(rm.features.duration[actions])
                    generated.append(actions.copy())
                    previous=actions.copy()
                    if mode=="append":
                        ctx=np.column_stack([ctx,actions]);seen[np.arange(n*g),actions]=True;counts=counts+1
                q=pd.DataFrame({k:states[k] for k in ["qid","uid","timestamp","all_positives"]})
                q["known_positives"]=states["positives"].sum(1)
                for key,sequence in values.items():
                    v=np.stack(sequence,axis=1).reshape(n,g,h)
                    per_draw=(v*discount[None,None,:]).sum(2)/discount.sum()
                    q[f"discounted_mean_{key}"]=per_draw.mean(1)
                    q[f"step1_{key}"]=v[:,:,0].mean(1)
                    q[f"step5_{key}"]=v[:,:,-1].mean(1)
                    if key in ["completion","audit_completion"]:
                        q[f"mc_se_{key}"]=per_draw.std(1,ddof=1)/np.sqrt(g)
                q.to_csv(target/f"{name}_{seed}_{mode}.csv.gz",index=False)
                frames.setdefault((name,mode),[]).append(q)
                record={"method":name,"seed":seed,"mode":mode,"initial_queries":n,"trajectories_per_query":g,
                    **{c:float(q[c].mean()) for c in q if c not in ["qid","uid","timestamp","all_positives","known_positives"]}}
                records.append(record)
                print(f"Session {name}/{seed}/{mode}: G5 completion={record['discounted_mean_completion']:.4f} audit={record['discounted_mean_audit_completion']:.4f}",flush=True)
            del model
    frame=pd.DataFrame(records)
    frame.to_csv(target/"per_seed.csv",index=False)
    frame.drop(columns="seed").groupby(["method","mode"]).mean().reset_index().to_csv(target/"summary.csv",index=False)
    averaged={key:mean_queries(v) for key,v in frames.items()}
    for (name,mode),q in averaged.items():
        q.to_csv(target/f"{name}_{mode}_mean.csv.gz",index=False)
    intervals=[]
    a,b=config["primary_comparison"]
    for mode in ["append","frozen"]:
        for metric in ["discounted_mean_completion","discounted_mean_audit_completion","discounted_mean_within_list_repeat"]:
            intervals.append({"baseline":a,"candidate":b,"mode":mode,**paired_bootstrap(averaged[(a,mode)],averaged[(b,mode)],metric,config["bootstrap_reps"])})
    dump(target/"paired_comparisons.json",intervals)
    dump(target/"protocol.json",{"initial_state_sample":"uniform 256 from test query sample, seed 622; no future-continuation filter",
        "queries":n,"query_indices":indices.tolist(),"horizon":h,"draws_per_query":g,"gamma":config["rl_gamma"],
        "aggregation":"sum gamma^t*r_t / sum gamma^t; first average trajectories, then seeds, then queries",
        "uniform_seed":623,"temperature":1.,"transitions":["append every sampled action","frozen real prefix"],
        "causal_semantics":"Both modes are surrogate assumptions; scores are predictions, not real retention or satisfaction.",
        "checkpoint_selection":"No model selection uses these final diagnostic test returns; checkpoints used first-step validation utility."})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/main.json")
    main(load_config(p.parse_args().config))
