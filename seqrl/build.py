"""Strict timestamp-group prefixes, train-only vocabulary, and censored outcomes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .common import dump,load_config,sha


def prepare_output(path):
    """Submission archives contain reference results, not reusable tensor caches."""
    out=Path(path)
    if (out/"ARCHIVE_RESULTS_ONLY.json").exists():
        index=1
        reference=out.with_name(f"reference_{out.name}_{index}")
        while reference.exists():
            index+=1
            reference=out.with_name(f"reference_{out.name}_{index}")
        out.rename(reference)
        print(f"Preserved supplied empirical results in {reference}; starting a fresh reproducible run",flush=True)
    out.mkdir(parents=True,exist_ok=True)
    return out


def prefix(tokens,times,cut,context):
    """Only records strictly earlier than this timestamp may enter the state."""
    stop=int(np.searchsorted(times,cut,side="left"))
    values=np.concatenate([np.array([1],dtype=np.int32),tokens[max(0,stop-context+1):stop]])
    x=np.zeros(context,dtype=np.int32)
    x[-len(values):]=values
    seen=np.unique(tokens[:stop])
    seen=seen[seen>=3]
    history_max=int(times[stop-1]) if stop else -1
    assert history_max < cut
    return x,seen,stop,history_max


def context_arrays(records,by_user,config):
    n=len(records)
    ctx=np.zeros((n,config["context"]),np.int32)
    seen=np.zeros((n,config["vocab_items"]+3),bool)
    counts=np.zeros(n,np.int32)
    hmax=np.zeros(n,np.int64)
    for i,row in enumerate(records.itertuples(index=False)):
        tokens,times=by_user[int(row.uid)]["tokens"],by_user[int(row.uid)]["times"]
        ctx[i],s,counts[i],hmax[i]=prefix(tokens,times,int(row.timestamp),config["context"])
        seen[i,s]=True
    assert (hmax<records.timestamp.to_numpy()).all()
    return {"contexts":ctx,"seen":seen,"history_count":counts,"history_max":hmax,
            "uid":records.uid.to_numpy().astype(np.int64),"timestamp":records.timestamp.to_numpy().astype(np.int64)}


def main(config):
    started=time.perf_counter()
    source,out=Path(config["data_dir"]),prepare_output(config["output_dir"])
    cache=out/"cache"
    cache.mkdir(parents=True,exist_ok=True)
    fp=out/"build_fingerprint.json"
    if fp.exists():
        previous=json.loads(fp.read_text())
        assert previous["config"]==config,"Changed config: use a new output_dir"
        assert all(sha(p)==value for p,value in previous["inputs"].items()),"Changed build inputs"
        required=["sft","catalog"]+[f"{kind}_{s}" for s in ["train","validation","confirmation","test"] for kind in ["states","reward"]]+[f"rollouts_{s}" for s in ["validation","confirmation","test"]]
        assert all((cache/f"{name}.npz").exists() for name in required),"Incomplete cache; remove build_fingerprint.json to rebuild"
        print("Reusing fixed temporal data cache",flush=True)
        return
    events=pd.read_parquet(source/"events.parquet")
    end=int(events.timestamp.max())+5
    test_start=end-config["test_days"]*86400
    confirm_start=test_start-config["confirmation_days"]*86400
    train_end=confirm_start-config["validation_days"]*86400
    bounds={"train":(0,train_end),"validation":(train_end,confirm_start),"confirmation":(confirm_start,test_start),"test":(test_start,end)}
    train=events[events.timestamp<train_end]
    frequency=train.groupby("item_id").size().rename("count").reset_index().sort_values(["count","item_id"],ascending=[False,True])
    ids=frequency.item_id.to_numpy()[:config["vocab_items"]]
    mapping={int(item):i+3 for i,item in enumerate(ids)}
    events["token"]=events.item_id.map(mapping).fillna(2).astype(np.int32)
    duration=np.zeros(config["vocab_items"]+3,np.float32)
    pop=np.zeros_like(duration)
    med=train.groupby("item_id").track_length_seconds.median()
    count=frequency.set_index("item_id")["count"]
    for item,token in mapping.items():
        duration[token]=float(med.loc[item])
        pop[token]=float(count.loc[item])
    np.savez_compressed(cache/"catalog.npz",item_ids=ids,duration=duration,popularity=pop)
    dump(out/"vocabulary.json",{"special_tokens":{"PAD":0,"BOS":1,"UNK":2},"items":list(map(int,ids)),
        "source":"top counts before train_end only","train_end":train_end})
    by_user={}
    query_rows=[]
    windows=[]
    diag_candidates={s:[] for s in ["validation","confirmation","test"]}
    window_width=config["context"]
    for uid,part in events.groupby("uid",sort=True):
        ts=part.timestamp.to_numpy().astype(np.int64)
        tok=part.token.to_numpy().astype(np.int32)
        organic=part.is_organic.to_numpy()
        raw_items=part.item_id.to_numpy()
        starts=np.r_[0,np.flatnonzero(np.diff(ts))+1]
        sizes=np.diff(np.r_[starts,len(ts)])
        size_per_event=np.repeat(sizes,sizes)
        ntrain=int(np.searchsorted(ts,train_end,side="left"))
        valid=(tok>=3)&(organic==0)&(size_per_event==1)&(ts<train_end)
        by_user[int(uid)]={"tokens":tok,"times":ts,"organic":organic,"valid":valid,"starts":starts,"sizes":sizes}
        for start in range(0,ntrain,max(1,window_width//2)):
            stop=min(start+window_width,ntrain)
            if valid[start:stop].any():
                windows.append((int(uid),start,stop))
        nonorg=np.add.reduceat((organic==0).astype(np.int32),starts)>0
        for gi in np.flatnonzero(nonorg):
            a,b=int(starts[gi]),int(starts[gi]+sizes[gi])
            t=int(ts[a])
            # Raw IDs distinguish multiple out-of-vocabulary positive items.
            raw_pos=np.unique(raw_items[a:b][organic[a:b]==0])
            known=sorted({mapping[int(item)] for item in raw_pos if int(item) in mapping})
            phase=next(s for s,(lo,hi) in bounds.items() if lo<=t<hi)
            query_rows.append({"uid":int(uid),"timestamp":t,"phase":phase,"positive_tokens":known,"all_positives":len(raw_pos),"group_index":int(gi)})
            if phase!="train" and gi+config["rl_horizon"]<=len(starts):
                future=starts[gi:gi+config["rl_horizon"]]
                if (sizes[gi:gi+config["rl_horizon"]]==1).all() and (organic[future]==0).all() and ts[future[-1]]<bounds[phase][1] and (np.diff(ts[future])<=1800).all():
                    diag_candidates[phase].append((int(uid),list(map(int,ts[future])),list(map(int,tok[future]))))
    print(f"Vocabulary {len(ids)}; temporal bounds {bounds}; building strict prefixes",flush=True)
    rng=np.random.default_rng(config["seed"])
    wi=rng.choice(len(windows),min(config["sft_windows"],len(windows)),replace=False)
    x=np.zeros((len(wi),window_width+1),np.int32)
    y=np.full_like(x,-100)
    valid_targets=0
    for i,j in enumerate(wi):
        uid,a,b=windows[j]
        row=by_user[uid]
        x[i,0]=1
        x[i,1:b-a+1]=row["tokens"][a:b]
        y[i,1:b-a+1]=np.where(row["valid"][a:b],row["tokens"][a:b],-100)
        valid_targets+=int(row["valid"][a:b].sum())
    np.savez_compressed(cache/"sft.npz",tokens=x,labels=y)
    query=pd.DataFrame(query_rows)
    query["qid"]=np.arange(len(query),dtype=np.int64)
    split_records={}
    for phase in bounds:
        candidates=query[query.phase==phase]
        if phase=="train":
            candidates=candidates[candidates.positive_tokens.map(len)>0]
            cap=config["rl_states"]
        else:
            cap=config[f"{phase}_queries"]
        sampled=candidates.iloc[np.sort(rng.choice(len(candidates),min(cap,len(candidates)),replace=False))].reset_index(drop=True)
        state=context_arrays(sampled,by_user,config)
        positives=np.zeros((len(sampled),config["vocab_items"]+3),bool)
        for i,values in enumerate(sampled.positive_tokens):
            positives[i,values]=True
        np.savez_compressed(cache/f"states_{phase}.npz",**state,positives=positives,
                            all_positives=sampled.all_positives.to_numpy().astype(np.int32),qid=sampled.qid.to_numpy())
        split_records[phase]={"start_inclusive":bounds[phase][0],"end_exclusive":bounds[phase][1],
            "available_queries":len(candidates),"sampled_queries":len(sampled),"sampled_users":int(sampled.uid.nunique()),
            "in_vocabulary_positive_coverage":float(positives.sum()/sampled.all_positives.sum()),
            "fully_oov_queries":int((positives.sum(1)==0).sum()),"strict_history":bool((state["history_max"]<state["timestamp"]).all())}
        print(f"{phase}: {len(sampled)} queries, target coverage {split_records[phase]['in_vocabulary_positive_coverage']:.3f}",flush=True)
        if phase!="train":
            ds=diag_candidates[phase]
            choose=rng.choice(len(ds),min(config["rollout_queries"],len(ds)),replace=False)
            chosen=[ds[i] for i in choose]
            teacher=[]
            for step in range(config["rl_horizon"]):
                r=pd.DataFrame([{"uid":u,"timestamp":times[step]} for u,times,_ in chosen])
                teacher.append(context_arrays(r,by_user,config))
            np.savez_compressed(cache/f"rollouts_{phase}.npz",**{k:np.stack([r[k] for r in teacher],axis=1) for k in teacher[0]},
                targets=np.array([items for _,_,items in chosen],dtype=np.int32))
    # One labelled row per (user,timestamp,item), preserving ambiguous repeated outcomes as their mean.
    r=events[(events.token>=3)&(events.is_organic==0)].copy()
    r["completion"]=np.clip(r.played_ratio_pct.to_numpy()/100,0,1)
    r["maturity_seconds"]=r.track_length_seconds.to_numpy()*r.played_ratio_pct.to_numpy()/100
    reward=r.groupby(["uid","timestamp","token"],sort=False).agg(completion=("completion","mean"),maturity_seconds=("maturity_seconds","max")).reset_index()
    reward_records={}
    return_start=config["reward_components"]["return_start_seconds"]
    return_stop=config["reward_components"]["return_stop_seconds"]
    uid_all=np.sort(events.uid.unique())
    part_a=set(np.random.default_rng(700).choice(uid_all,len(uid_all)//2,replace=False).tolist())
    pd.DataFrame({"uid":uid_all,"partition":["optimizer" if int(u) in part_a else "auditor" for u in uid_all]}).to_csv(out/"reward_user_partition.csv",index=False)
    for phase,(lo,hi) in bounds.items():
        valid=reward[(reward.timestamp>=lo)&(reward.timestamp+return_stop<hi)&(reward.timestamp+reward.maturity_seconds<hi)]
        cap=config[f"reward_{phase}_rows"]
        subset=valid.iloc[np.sort(rng.choice(len(valid),min(cap,len(valid)),replace=False))].reset_index(drop=True)
        state=context_arrays(subset,by_user,config)
        returns=np.zeros(len(subset),np.float32)
        for i,row in enumerate(subset.itertuples(index=False)):
            times=by_user[int(row.uid)]["times"]
            returns[i]=int(np.searchsorted(times,int(row.timestamp)+return_stop,side="right")>np.searchsorted(times,int(row.timestamp)+return_start,side="right"))
        np.savez_compressed(cache/f"reward_{phase}.npz",**state,actions=subset.token.to_numpy().astype(np.int32),
            completion=subset.completion.to_numpy().astype(np.float32),return_proxy=returns,
            optimizer_users=np.array([int(u) in part_a for u in subset.uid],dtype=bool))
        reward_records[phase]={"rows":len(subset),"eligible_rows":len(valid),"optimizer_rows":int(sum(int(u) in part_a for u in subset.uid)),
                              "return_proxy_rate":float(returns.mean()),"strict_history":bool((state["history_max"]<state["timestamp"]).all()),
                              "all_outcomes_observable_inside_split":bool((subset.timestamp+return_stop<hi).all())}
        print(f"Reward {phase}: {len(subset)} rows, return proxy mean {returns.mean():.3f}",flush=True)
    audit={"config":config,"source_stats":json.loads((source/"stats.json").read_text()),"splits":split_records,
           "reward_splits":reward_records,"sft_windows":len(wi),"sft_unambiguous_target_tokens":valid_targets,
           "sft_context_only_unknown_fraction":float((x==2).sum()/np.maximum((x!=0).sum(),1)),
           "timestamp_units":"seconds rounded to 5 seconds, not 5-second counts",
           "ties_policy":"Canonical order only inside past blocks; LM target losses masked for every tied timestamp group; query prefix strictly less than target time.",
           "build_seconds":time.perf_counter()-started}
    dump(out/"audit.json",audit)
    fingerprint={"config":config,"inputs":{str(p):sha(p) for p in [source/"events.parquet",out/"vocabulary.json",Path("data/SOURCE_MANIFEST.json")]}}
    fp=out/"build_fingerprint.json"
    if fp.exists() and json.loads(fp.read_text())!=fingerprint:
        raise ValueError("Changed data/config: use a new output_dir")
    dump(fp,fingerprint)
    print(f"Build finished in {audit['build_seconds']:.1f}s",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/main.json")
    main(load_config(parser.parse_args().config))
