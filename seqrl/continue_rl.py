"""Optional scheduler for sequential RL seeds alongside ongoing SFT replication.

Only one RL trainer writes records at a time. Waits for the preceding seed's
complete records and each SFT seed's final metadata, never an interim .pt.
"""
import argparse
import json
from pathlib import Path
import time
from .common import load_config
from .rl import train


def ready(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None


def main(config):
    out=Path(config["output_dir"])
    previous=config["model_seeds"][0]
    for seed in config["model_seeds"][1:]:
        while True:
            selection=ready(out/"sft_selection.json")
            records=ready(out/"rl_records.json") or {}
            complete=all(f"{v['name']}_{previous}" in records and Path(records[f"{v['name']}_{previous}"]["path"]).exists() for v in config["variants"])
            meta=None if selection is None else ready(Path(selection["models"][str(seed)]).with_suffix(".json"))
            if complete and meta is not None:
                break
            time.sleep(5)
        print(f"Prior RL seed {previous} and final SFT seed {seed} complete; starting RL seed {seed}",flush=True)
        train(config,seeds=[seed])
        previous=seed


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/main.json")
    main(load_config(p.parse_args().config))
