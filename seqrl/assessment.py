"""Exact full-catalog ranking and common-random-number reward estimates."""
import numpy as np
import pandas as pd
from scipy.special import softmax,logsumexp

from .metrics import per_query,aggregate,behavior
from .reward import utility


def assess(config,states,scores,reward,objective="completion",reference_scores=None,audit=True):
    q,order,p=per_query(states,scores)
    n,m=p.shape
    samples=config["evaluation_action_samples"]
    # Same U values for every policy and seed: controls evaluation Monte Carlo variance.
    u=np.random.default_rng(901).random((n,samples))
    cdf=np.cumsum(p.astype(np.float64),axis=1)
    cdf/=cdf[:,-1:]
    actions=np.vstack([np.searchsorted(cdf[i],u[i],side="right") for i in range(n)]).astype(np.int32)
    actions=np.minimum(actions,m-1)
    assert (actions>=3).all()
    profiles=reward.features.profiles(states["contexts"],states["seen"],states["history_count"])
    rows=np.repeat(np.arange(n),samples)
    flat=actions.ravel()
    preds=reward.predict_matrix(reward.features.actions(profiles,flat,rows),audit=audit)
    objective_values=utility(preds,flat,states["contexts"][rows],reward.features.catalog,objective)
    for key,value in preds.items():
        q[f"expected_{key}"]=np.asarray(value).reshape(n,samples).mean(1)
    q["expected_utility"]=objective_values.reshape(n,samples).mean(1)
    q["reward_mc_standard_error"]=np.asarray(preds["completion"]).reshape(n,samples).std(1,ddof=1)/np.sqrt(samples)
    q["expected_duration"]=reward.features.duration[actions].mean(1)
    q["greedy_duration"]=reward.features.duration[order[:,0]]
    q["expected_rm_disagreement"]=np.abs(preds["completion"]-preds.get("audit_completion",preds["completion"])).reshape(n,samples).mean(1)
    from .sft import read_npz
    from pathlib import Path
    support=read_npz(Path(config["output_dir"])/"cache/reward_baselines.npz")["action_count"]
    q["prob_no_optimizer_rm_item_labels"]=(p*(support==0)[None,:]).sum(1)
    q["prob_below5_optimizer_rm_item_labels"]=(p*(support<5)[None,:]).sum(1)
    q["greedy_optimizer_rm_item_labels"]=support[order[:,0]]
    if reference_scores is not None:
        logp=scores-logsumexp(scores,axis=1,keepdims=True)
        logr=reference_scores-logsumexp(reference_scores,axis=1,keepdims=True)
        q["kl_to_sft"]=(p*(logp-logr)).sum(1)
    metric={**aggregate(q),**behavior(states,order,p,reward.features.catalog),"queries":len(q),
            "reward_action_samples_per_query":samples,"ranking_catalog_items":config["vocab_items"]}
    return metric,q,order[:,:20]
