"""Audit source completeness, temporal availability, LM masks and sealed results."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .build import prefix
from .common import dump,load_config,sha
from .sft import read_npz


def main(config,final=False):
    out=Path(config["output_dir"])
    events=pd.read_parquet(Path(config["data_dir"])/"events.parquet")
    vocabulary=json.loads((out/"vocabulary.json").read_text())
    mapping={item:i+3 for i,item in enumerate(vocabulary["items"])}
    events["token"]=events.item_id.map(mapping).fillna(2).astype(np.int32)
    by_user={int(u):p for u,p in events.groupby("uid",sort=True)}
    source=json.loads(Path("data/SOURCE_MANIFEST.json").read_text())
    assert sha("data/raw/listens.parquet")==source["sha256"]
    parquet=pq.ParquetFile("data/raw/listens.parquet")
    population=np.sort(parquet.read(columns=["uid"]).column("uid").to_numpy())
    expected=np.sort(np.random.default_rng(config["seed"]).choice(population,len(by_user),replace=False))
    assert np.array_equal(expected,np.sort(list(by_user)))
    fields=["timestamp","item_id","is_organic","played_ratio_pct","track_length_seconds"]
    verified_users=0
    for batch in parquet.iter_batches(batch_size=64):
        df=batch.to_pandas()
        for row in df[df.uid.isin(expected)].itertuples(index=False):
            values={k:np.asarray(getattr(row,k)) for k in fields}
            order=np.lexsort((np.arange(len(values["timestamp"])),values["item_id"],values["timestamp"]))
            current=by_user[int(row.uid)]
            assert len(current)==len(order)
            for k in fields:
                assert np.array_equal(current[k].to_numpy(),values[k][order]),(row.uid,k)
            verified_users+=1
    assert verified_users==len(expected)
    train=events[events.timestamp<vocabulary["train_end"]]
    counts=train.groupby("item_id").size().rename("n").reset_index().sort_values(["n","item_id"],ascending=[False,True])
    assert counts.item_id.iloc[:config["vocab_items"]].tolist()==vocabulary["items"]
    windows=[]
    for uid,p in by_user.items():
        ts=p.timestamp.to_numpy();tok=p.token.to_numpy()
        sizes=p.groupby("timestamp").timestamp.transform("size").to_numpy()
        valid=(tok>=3)&(p.is_organic.to_numpy()==0)&(sizes==1)&(ts<vocabulary["train_end"])
        ntrain=int(np.searchsorted(ts,vocabulary["train_end"]))
        for a in range(0,ntrain,config["context"]//2):
            b=min(a+config["context"],ntrain)
            if valid[a:b].any():
                windows.append((uid,a,b))
        p["valid_label"]=valid
    wi=np.random.default_rng(config["seed"]).choice(len(windows),min(config["sft_windows"],len(windows)),replace=False)
    sft=read_npz(out/"cache/sft.npz")
    for i,j in enumerate(wi):
        u,a,b=windows[j];p=by_user[u]
        x=np.zeros(config["context"]+1,np.int32);x[0]=1;x[1:b-a+1]=p.token.to_numpy()[a:b]
        y=np.full_like(x,-100);y[1:b-a+1]=np.where(p.valid_label.to_numpy()[a:b],p.token.to_numpy()[a:b],-100)
        assert np.array_equal(x,sft["tokens"][i]) and np.array_equal(y,sft["labels"][i]),"LM mask mismatch"
    audit=json.loads((out/"audit.json").read_text())
    checks=[]
    for split,bound in audit["splits"].items():
        for kind in ["states","reward"]:
            z=read_npz(out/"cache"/f"{kind}_{split}.npz")
            assert (z["history_max"]<z["timestamp"]).all()
            assert ((z["timestamp"]>=bound["start_inclusive"])&(z["timestamp"]<bound["end_exclusive"])).all()
            indices=np.random.default_rng(70).choice(len(z["contexts"]),min(256,len(z["contexts"])),replace=False)
            if kind=="reward":
                assert (z["timestamp"]+config["reward_components"]["return_stop_seconds"]<bound["end_exclusive"]).all()
            for i in indices:
                p=by_user[int(z["uid"][i])]
                ts=p.timestamp.to_numpy();tokens=p.token.to_numpy();cut=int(z["timestamp"][i])
                x,seen,count,last=prefix(tokens,ts,cut,config["context"])
                assert np.array_equal(x,z["contexts"][i]) and count==z["history_count"][i] and last==z["history_max"][i]
                assert np.array_equal(seen,np.flatnonzero(z["seen"][i]))
                target=p[(p.timestamp==cut)&(p.is_organic==0)]
                if kind=="reward":
                    current=target[target.token==z["actions"][i]]
                    completion=np.clip(current.played_ratio_pct.to_numpy()/100,0,1).mean()
                    assert abs(completion-z["completion"][i])<1e-6
                    maturity=(current.track_length_seconds*current.played_ratio_pct/100).max()
                    assert cut+maturity<bound["end_exclusive"]
                    observed=int(np.searchsorted(ts,cut+172800,side="right")>np.searchsorted(ts,cut+86400,side="right"))
                    assert observed==z["return_proxy"][i]
                else:
                    known=np.unique(target.token.to_numpy());known=known[known>=3]
                    assert np.array_equal(known,np.flatnonzero(z["positives"][i]))
                    assert target.item_id.nunique()==z["all_positives"][i]
            checks.append({"split":split,"kind":kind,"all_rows_strict_history":len(z["contexts"]),"exact_reconstructed_rows":len(indices)})
            del z;gc.collect()
    result={"source_complete_users":verified_users,"source_complete_events":len(events),"source_sha256":source["sha256"],
        "uniform_user_sample_verified":True,"train_only_vocabulary_verified":True,"all_lm_windows_reconstructed":len(wi),
        "all_lm_labels_nonorganic_known_and_single_timestamp":True,"cached_prefix_checks":checks}
    if final:
        lock=json.loads((out/"selection.lock.json").read_text())
        assert all(sha(p)==v for p,v in lock["model_sha256"].items())
        assert sha(out/"evaluation/confirmation/summary.json")==lock["confirmation_summary_sha256"]
        assert all(sha(p)==value for p,value in lock["evaluation_source_sha256"].items())
        assert sha("EXPERIMENT_PLAN.md")==lock["protocol_plan_sha256"] and sha("DIAGNOSTIC_PLAN.md")==lock["diagnostic_plan_sha256"]
        models=json.loads((out/"reward_models.json").read_text())
        assert all(sha(out/p)==v for p,v in models["files_sha256"].items())
        summary=json.loads((out/"evaluation/test/summary.json").read_text())
        for name,stats in summary.items():
            q=pd.read_csv(out/"evaluation/test"/f"{name}_mean.csv.gz")
            for metric in ["ndcg@10","recall@10","map@10","expected_completion","expected_audit_completion"]:
                assert abs(q[metric].mean()-stats["mean"][metric])<1e-7
        result.update({"sealed_model_hashes_verified":len(lock["model_sha256"]),"saved_test_means_recomputed":True,
            "selected_method":lock["selected_method"],"test_metrics_before_seal":lock["test_metrics_computed_before_seal"]})
    dump(out/("validation_final.json" if final else "validation_data.json"),result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/main.json");p.add_argument("--final",action="store_true")
    args=p.parse_args();main(load_config(args.config),args.final)
