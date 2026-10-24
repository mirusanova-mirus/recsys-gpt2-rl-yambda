"""Full action-catalog ranking, behavior diagnostics and paired user bootstrap."""
import numpy as np
import pandas as pd
from scipy.special import softmax


def per_query(states,scores):
    order=np.argsort(-scores,axis=1,kind="stable")
    pos=states["positives"]
    n=len(scores)
    out={"qid":states["qid"],"uid":states["uid"],"timestamp":states["timestamp"],
         "all_positives":states["all_positives"],"known_positives":pos.sum(1)}
    ranked=np.take_along_axis(pos,order,axis=1)
    for k in [5,10,20]:
        discounts=1/np.log2(np.arange(1,k+1)+1)
        hits=ranked[:,:k].sum(1)
        dcg=(ranked[:,:k]*discounts).sum(1)
        idcg=np.r_[0,np.cumsum(discounts)]
        all_count=states["all_positives"]
        known=pos.sum(1)
        out[f"recall@{k}"]=hits/np.maximum(all_count,1)
        out[f"ndcg@{k}"]=dcg/idcg[np.minimum(k,all_count)]
        precision=np.cumsum(ranked[:,:k],axis=1)/np.arange(1,k+1)
        ap=(precision*ranked[:,:k]).sum(1)
        out[f"map@{k}"]=ap/np.minimum(k,all_count)
        out[f"conditional_recall@{k}"]=np.divide(hits,known,out=np.full(n,np.nan),where=known>0)
        out[f"conditional_ndcg@{k}"]=np.divide(dcg,idcg[np.minimum(k,known)],out=np.full(n,np.nan),where=known>0)
        out[f"conditional_map@{k}"]=np.divide(ap,np.minimum(k,known),out=np.full(n,np.nan),where=known>0)
    p=softmax(scores,axis=1)
    out["policy_entropy"]=-np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-30)),0),axis=1)
    out["prob_history_repeat"]=(p*states["seen"]).sum(1)
    out["top10_history_repeat"]=np.take_along_axis(states["seen"],order[:,:10],axis=1).mean(1)
    return pd.DataFrame(out),order,p


def behavior(states,order,p,catalog):
    counts=np.bincount(order[:,:10].ravel(),minlength=p.shape[1])[3:]
    x=np.sort(counts)
    gini=float((2*np.sum((np.arange(len(x))+1)*x)/(len(x)*x.sum()))-(len(x)+1)/len(x))
    top=np.bincount(order[:,0],minlength=p.shape[1])[3:]
    unique_users,first=np.unique(states["uid"],return_index=True)
    pairs=[]
    rng=np.random.default_rng(2026)
    if len(first)>1:
        for _ in range(min(2000,len(first)*2)):
            a,b=rng.choice(first,2,replace=False)
            inter=len(set(order[a,:10]).intersection(order[b,:10]))
            pairs.append(inter/(20-inter))
    return {"catalog_coverage@10":float((counts>0).mean()),"exposure_gini@10":gini,
        "top1_largest_item_share":float(top.max()/len(order)),"cross_user_jaccard@10":float(np.mean(pairs)) if pairs else None,
        "expected_track_seconds":float((p*catalog["duration"]).sum(1).mean()),
        "greedy_track_seconds":float(catalog["duration"][order[:,0]].mean())}


def aggregate(q):
    return {c:float(q[c].mean()) for c in q if c not in ["qid","uid","timestamp","all_positives","known_positives"]}


def paired_bootstrap(a,b,metric,reps=2000):
    keys=["qid","uid"]
    assert a[keys].equals(b[keys]),"Expected identical query ordering and users"
    delta=b[metric].to_numpy()-a[metric].to_numpy()
    good=np.isfinite(delta)
    frame=pd.DataFrame({"uid":a.uid.to_numpy()[good],"delta":delta[good]})
    grouped=frame.groupby("uid").delta.agg(["sum","size"])
    sums,counts=grouped["sum"].to_numpy(),grouped["size"].to_numpy()
    rng=np.random.default_rng(2026)
    values=[]
    for _ in range(reps):
        idx=rng.integers(0,len(grouped),len(grouped))
        values.append(sums[idx].sum()/counts[idx].sum())
    ci=np.quantile(values,[.025,.975])
    return {"metric":metric,"delta":float(np.mean(delta[good])),"ci95":list(map(float,ci)),"cluster":"uid",
        "users":len(grouped),"queries":int(good.sum()),"replicates":reps,"training_seed_handling":"mean of per-query metrics over the same 3 seeds, no probability ensemble"}
