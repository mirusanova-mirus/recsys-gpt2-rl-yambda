"""Optional scheduler: fit rewards/RL seed 42 while SFT replication seeds run.

This changes scheduling only. It waits for COMPLETE architecture-grid metadata,
never reads a partially trained checkpoint, and leaves normal make reproduce
sequential. Start only alongside one already-running seqrl.sft process.
"""
import argparse
import json
from pathlib import Path
import time
from .common import dump,load_config
from .reward import fit,quality
from .rl import train


def main(config):
    out=Path(config["output_dir"])
    seed=config["model_seeds"][0]
    paths=[out/"models"/f"sft_w{w}_{seed}.json" for w in config["sft_width_grid"]]
    while not all(p.exists() for p in paths):
        time.sleep(5)
    metadata=[json.loads(p.read_text()) for p in paths]
    grid=[(m["best_validation_ndcg@10"],w) for m,w in zip(metadata,config["sft_width_grid"])]
    _,width=max(grid,key=lambda pair:(pair[0],-pair[1]))
    selected={"width":width,"models":{str(s):str(out/"models"/f"sft_w{width}_{s}.pt") for s in config["model_seeds"]},
        "selection_metric":"validation query-macro NDCG@10 including OOV targets","grid":grid}
    dump(out/"sft_selection.json",selected)
    print(f"Completed SFT grid: selected width={width}; reward fit can overlap replication seeds",flush=True)
    fit(config)
    print(json.dumps(quality(config,"validation"),indent=2),flush=True)
    train(config,seeds=[seed])


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/main.json")
    main(load_config(p.parse_args().config))
