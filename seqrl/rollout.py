"""Five-step transition stress test, with actual future items as diagnostic anchors."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax,logsumexp

from .common import dump,load_config,seed_all,sha
from .evaluate import method_models
from .model import load_model,predict
from .reward import RewardModels
from .sft import read_npz


def run(config):
    out=Path(config["output_dir"])
    lock=json.loads((out/"selection.lock.json").read_text())
    seed_all(config["seed"],config["threads"])
    z=read_npz(out/"cache/rollouts_test.npz")
    methods=method_models(config)
    rm=RewardModels(config)
    records=[]
    rows=[]
    n,h=z["targets"].shape
    for name in methods:
        for seed,path in methods[name].items():
            assert sha(path)==lock["model_sha256"][path]
            model,_=load_model(path)
            teacher_scores=[]
            for mode in ["teacher_forced","append_all","frozen","model_acceptance"]:
                contexts=z["contexts"][:,0].copy()
                seen=z["seen"][:,0].copy()
                counts=z["history_count"][:,0].copy()
                generated=[]
                for step in range(h):
                    if mode=="teacher_forced":
                        contexts=z["contexts"][:,step]
                        seen=z["seen"][:,step]
                        counts=z["history_count"][:,step]
                    scores=predict(model,contexts)
                    order=np.argsort(-scores,axis=1,kind="stable")[:,:10]
                    actions=order[:,0]
                    target=z["targets"][:,step]
                    rank_hit=order==target[:,None]
                    ndcg=(rank_hit/np.log2(np.arange(10)+2)).sum(1)*(target>=3)
                    pred=rm.predict(contexts[:,-config["context"]:],seen,counts,actions,audit=True)
                    p=softmax(scores,axis=1)
                    entropy=-(p*np.log(np.maximum(p,1e-30))).sum(1)
                    if mode=="teacher_forced":
                        teacher_scores.append(scores.copy())
                    reference=teacher_scores[step]
                    kl=(p*(scores-logsumexp(scores,axis=1,keepdims=True)-reference+logsumexp(reference,axis=1,keepdims=True))).sum(1)
                    duplicate=np.zeros(n,bool) if not generated else (np.column_stack(generated)==actions[:,None]).any(1)
                    generated.append(actions.copy())
                    # Acceptance is a model assumption, not an observed interaction.
                    u=np.random.default_rng(8300+step).random(n)
                    accept=u<pred["completion"]
                    record={"method":name,"seed":seed,"mode":mode,"step":step+1,"queries":n,
                        "ndcg@10_anchor":float(ndcg.mean()),"recall@10_anchor":float((rank_hit.any(1)&(target>=3)).mean()),
                        "target_vocabulary_coverage":float((target>=3).mean()),
                        "optimizer_greedy_completion":float(pred["completion"].mean()),"auditor_greedy_completion":float(pred["audit_completion"].mean()),
                        "optimizer_return_proxy":float(pred["return_proxy"].mean()),"policy_entropy":float(entropy.mean()),
                        "same_as_teacher_top1":float((actions==reference.argmax(1)).mean()),"kl_to_teacher_context":float(kl.mean()),
                        "history_repeat":float(seen[np.arange(n),actions].mean()),"within_list_repeat":float(duplicate.mean()),
                        "model_acceptance_rate":float(accept.mean()),"track_seconds":float(rm.features.duration[actions].mean())}
                    records.append(record)
                    for i in range(n):
                        rows.append({"method":name,"seed":seed,"mode":mode,"step":step+1,"case":i,"uid":int(z["uid"][i,0]),
                            "target":int(target[i]),"recommendation":int(actions[i]),"ndcg@10_anchor":float(ndcg[i]),
                            "optimizer_completion":float(pred["completion"][i]),"auditor_completion":float(pred["audit_completion"][i]),
                            "kl_to_teacher_context":float(kl[i]),"within_list_repeat":int(duplicate[i])})
                    if mode in ["append_all","model_acceptance"]:
                        append=np.ones(n,bool) if mode=="append_all" else accept
                        # Left padding preserves valid position IDs for rejected actions.
                        expanded=np.zeros((n,contexts.shape[1]+1),np.int32)
                        expanded[append,:-1]=contexts[append]
                        expanded[append,-1]=actions[append]
                        expanded[~append,1:]=contexts[~append]
                        contexts=expanded
                        seen[np.flatnonzero(append),actions[append]]=True
                        counts=counts+append
                    del scores,p,order
                print(f"Rollout {name}/{seed}/{mode} complete",flush=True)
            del model,teacher_scores
            gc.collect()
    target_dir=out/"evaluation/rollouts"
    target_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(target_dir/"per_case.csv.gz",index=False)
    frame=pd.DataFrame(records)
    frame.to_csv(target_dir/"per_seed.csv",index=False)
    frame.drop(columns="seed").groupby(["method","mode","step"]).mean().reset_index().to_csv(target_dir/"summary.csv",index=False)
    dump(target_dir/"protocol.json",{"queries":n,"horizon":h,"source":"heldout single-event nonorganic runs; consecutive gaps <=1800 seconds",
        "semantics":"Actual future items are diagnostic anchors under alternative contexts; they are NOT counterfactual user choices.",
        "acceptance":"Common uniforms, accept if U < optimized RM completion. Heuristic, not validated transition model.",
        "modes":["teacher_forced","append_all","frozen","model_acceptance"]})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/main.json")
    run(load_config(p.parse_args().config))
