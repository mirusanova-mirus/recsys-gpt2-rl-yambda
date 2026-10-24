"""Print a real heldout prefix's recommendations from a sealed fitted model."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import load_config,seed_all
from .evaluate import method_models
from .model import load_model,predict
from .reward import RewardModels
from .sft import read_npz


def main(args):
    config=load_config(args.config);seed_all(314,config["threads"])
    out=Path(config["output_dir"])
    lock=json.loads((out/"selection.lock.json").read_text())
    name=args.method or lock["selected_method"]
    z=read_npz(out/"cache/states_test.npz")
    idx=0 if args.uid is None else int(np.flatnonzero(z["uid"]==args.uid)[0])
    model,_=load_model(method_models(config)[name][args.seed])
    ctx=z["contexts"][idx:idx+1]
    scores=predict(model,ctx);actions=np.argsort(-scores[0],kind="stable")[:10]
    rm=RewardModels(config)
    p=rm.predict(np.repeat(ctx,10,axis=0),np.repeat(z["seen"][idx:idx+1],10,axis=0),np.repeat(z["history_count"][idx],10),actions,True)
    ids=np.r_[[-1,-1,-1],rm.features.catalog["item_ids"]]
    print(f"uid={z['uid'][idx]}, timestamp={z['timestamp'][idx]}, policy={name}, seed={args.seed}")
    print(pd.DataFrame({"rank":np.arange(1,11),"item_id":ids[actions],"predicted_completion":p["completion"],
        "auditor_completion":p["audit_completion"],"duration_seconds":rm.features.duration[actions],
        "previously_seen":z["seen"][idx,actions]}).to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/main.json")
    p.add_argument("--method");p.add_argument("--seed",type=int,default=42);p.add_argument("--uid",type=int)
    main(p.parse_args())
