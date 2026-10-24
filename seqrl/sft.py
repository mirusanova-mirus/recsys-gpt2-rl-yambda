"""Supervised next-item GPT-2 training and validation-only architecture selection."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

from .common import dump,load_config,seed_all,sha
from .metrics import aggregate,per_query
from .model import create_model,lm_loss,load_model,predict,save_model


def read_npz(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def main(config):
    out=Path(config["output_dir"])
    cache=out/"cache"
    models=out/"models"
    models.mkdir(exist_ok=True)
    train=read_npz(cache/"sft.npz")
    validation=read_npz(cache/"states_validation.npz")
    fingerprint={"config":config,"vocabulary_sha256":sha(out/"vocabulary.json"),
        "train_array_sha256":__import__("hashlib").sha256(train["tokens"].tobytes()+train["labels"].tobytes()).hexdigest(),
        "source_versions":{p:importlib.metadata.version(p) for p in ["torch","transformers"]}}
    fp=out/"training_fingerprint.json"
    if fp.exists() and json.loads(fp.read_text())!=fingerprint:
        raise ValueError("Changed inputs or config; use a new output_dir")
    dump(fp,fingerprint)
    records={}
    started=time.perf_counter()

    def fit(width,seed,name):
        path=models/f"{name}.pt"
        meta_path=path.with_suffix(".json")
        signature={"width":width,"seed":seed,"training_fingerprint":fingerprint}
        if path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text())
            assert meta["signature"]==signature
            records[name]=meta
            return meta
        seed_all(seed,config["threads"])
        model=create_model(config,width)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config["sft_lr"],weight_decay=.01)
        rng=np.random.default_rng(seed)
        best,bad,curve=-1.0,0,[]
        train_seconds=0.0
        for epoch in range(1,config["sft_epochs"]+1):
            model.train()
            tick=time.perf_counter()
            total,tokens=0.,0
            indices=rng.permutation(len(train["tokens"]))
            for i in range(0,len(indices),config["sft_batch"]):
                batch=indices[i:i+config["sft_batch"]]
                x=torch.as_tensor(train["tokens"][batch],dtype=torch.long)
                y=torch.as_tensor(train["labels"][batch],dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                loss,n=lm_loss(model,x,y)
                if loss is None:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                total+=float(loss.detach())*n
                tokens+=n
            elapsed=time.perf_counter()-tick
            train_seconds+=elapsed
            q,_,_=per_query(validation,predict(model,validation["contexts"]))
            metric=aggregate(q)
            entry={"epoch":epoch,"loss":total/max(tokens,1),"target_tokens":tokens,"fit_seconds":elapsed,**metric}
            curve.append(entry)
            print(f"SFT {name} epoch={epoch} loss={entry['loss']:.4f} valNDCG10={metric['ndcg@10']:.5f} fit={elapsed:.1f}s",flush=True)
            if metric["ndcg@10"]>best+1e-8:
                best=metric["ndcg@10"]
                bad=0
                save_model(path,model,{"seed":seed,"width":width,"epoch":epoch,"validation_ndcg@10":best})
            else:
                bad+=1
            dump(out/f"curve_{name}.json",curve)
            if bad>=config["sft_patience"]:
                break
        meta={"signature":signature,"best_validation_ndcg@10":best,"fit_seconds":train_seconds,"curve":curve,
            "parameters":sum(p.numel() for p in model.parameters()),"model_bytes":path.stat().st_size,"model_sha256":sha(path)}
        dump(meta_path,meta)
        records[name]=meta
        del model,optimizer
        gc.collect()
        return meta

    grid=[]
    for width in config["sft_width_grid"]:
        m=fit(width,config["model_seeds"][0],f"sft_w{width}_42")
        grid.append((m["best_validation_ndcg@10"],width))
    _,width=max(grid,key=lambda pair:(pair[0],-pair[1]))
    final={str(seed):str(models/f"sft_w{width}_{seed}.pt") for seed in config["model_seeds"]}
    # Reward fitting needs only the selected seed-42 embedding, so the architecture
    # decision can be exposed before the independent replication seeds finish.
    dump(out/"sft_selection.json",{"width":width,"models":final,"selection_metric":"validation query-macro NDCG@10 including OOV targets","grid":grid})
    for seed in config["model_seeds"]:
        name=f"sft_w{width}_{seed}"
        fit(width,seed,name)
        final[str(seed)]=str(models/f"{name}.pt")
    dump(out/"sft_selection.json",{"width":width,"models":final,"selection_metric":"validation query-macro NDCG@10 including OOV targets","grid":grid})
    dump(out/"sft_records.json",records)
    dump(out/"environment_sft.json",{"platform":platform.platform(),"python":__import__("sys").version,
        "packages":{p:importlib.metadata.version(p) for p in ["torch","transformers","numpy","pandas","catboost","scikit-learn","pyarrow"]},
        "threads":config["threads"],"wall_seconds":time.perf_counter()-started,
        "fit_seconds":sum(r["fit_seconds"] for r in records.values()),"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20})
    print(f"Selected GPT-2 width={width}; SFT models ready",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/main.json")
    main(load_config(parser.parse_args().config))
