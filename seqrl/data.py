"""Pinned public Yambda download and complete-user-history extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
REMOTE = "sequential/50m/listens.parquet"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "listens.parquet"
    url = f"https://huggingface.co/datasets/yandex/yambda/resolve/{REVISION}/{REMOTE}"
    if not target.exists():
        partial = target.with_suffix(".part")
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=(30,120)) as r:
                    r.raise_for_status()
                    written, tick = 0, time.perf_counter()
                    with partial.open("wb") as f:
                        for chunk in r.iter_content(4*1024*1024):
                            f.write(chunk)
                            written += len(chunk)
                            if written // (64*1024*1024) > (written-len(chunk)) // (64*1024*1024):
                                print(f"Downloaded {written/2**20:.0f} MiB in {time.perf_counter()-tick:.1f}s",flush=True)
                partial.replace(target)
                break
            except requests.RequestException:
                if attempt==2:
                    raise
                time.sleep(2)
    digest = sha256(target)
    record = {"repository":"https://huggingface.co/datasets/yandex/yambda", "revision":REVISION,
              "remote_path":REMOTE,"url":url,"sha256":digest,"bytes":target.stat().st_size}
    known = Path("data/SOURCE_MANIFEST.json")
    if known.exists():
        previous = json.loads(known.read_text())
        if previous["revision"] == REVISION and previous["sha256"] != digest:
            raise ValueError("Pinned source checksum mismatch")
    known.parent.mkdir(parents=True,exist_ok=True)
    known.write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)
    return target


def prepare(path, out, n_users, seed):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"stats.json").exists() and (out/"events.parquet").exists():
        previous=json.loads((out/"stats.json").read_text())
        assert previous["sample_users"]==n_users and previous["sample_seed"]==seed,"Use another data directory for a different sample"
        assert previous["source"]["sha256"]==sha256(path),"Source changed"
        print("Reusing complete sampled user histories",flush=True)
        return
    parquet = pq.ParquetFile(path)
    print(parquet.schema_arrow,flush=True)
    ids = np.sort(parquet.read(columns=["uid"]).column("uid").to_numpy())
    assert len(ids)==len(np.unique(ids)), "Expected one sequential row per user"
    selected = np.sort(np.random.default_rng(seed).choice(ids,min(n_users,len(ids)),replace=False))
    chunks=[]
    for batch in parquet.iter_batches(batch_size=128):
        frame=batch.to_pandas()
        frame=frame[frame.uid.isin(selected)]
        for row in frame.itertuples(index=False):
            arrays={c:np.asarray(getattr(row,c)) for c in frame.columns if c!="uid"}
            if not arrays:
                continue
            lengths={len(a) for a in arrays.values()}
            assert len(lengths)==1
            piece=pd.DataFrame(arrays).rename(columns={"item_ids":"item_id","timestamps":"timestamp"})
            piece["uid"]=row.uid
            chunks.append(piece)
    events=pd.concat(chunks,ignore_index=True)
    # Preserve every provided record; stable within-timestamp order is only a convention.
    events=events.sort_values(["uid","timestamp","item_id"],kind="stable").reset_index(drop=True)
    events.to_parquet(out/"events.parquet",index=False)
    pd.Series(selected,name="uid").to_csv(out/"sampled_users.csv",index=False)
    gaps=events.groupby("uid").timestamp.diff().dropna()
    positive=gaps[gaps>0]
    record={"population_users":len(ids),"sample_users":len(selected),"sample_seed":seed,
        "events":len(events),"items":int(events.item_id.nunique()),
        "timestamp_min":int(events.timestamp.min()),"timestamp_max":int(events.timestamp.max()),
        "positive_gap_quantiles":positive.quantile([0,.1,.5,.9,.99,1]).to_dict(),
        "gap_gcd":int(np.gcd.reduce(positive.astype(np.int64).to_numpy())),
        "history_quantiles":events.groupby("uid").size().quantile([0,.25,.5,.75,.9,.99,1]).to_dict(),
        "same_timestamp_records":int(events.duplicated(["uid","timestamp"]).sum()),
        "duplicate_full_records":int(events.duplicated().sum()),
        "ratio_quantiles":events.played_ratio_pct.quantile([0,.1,.5,.9,.99,1]).to_dict(),
        "duration_quantiles":events.track_length_seconds.quantile([0,.1,.5,.9,.99,1]).to_dict(),
        "organic_fraction":float(events.is_organic.mean()),"source":json.loads(Path("data/SOURCE_MANIFEST.json").read_text())}
    (out/"stats.json").write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!="source"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--users",type=int,default=1000)
    parser.add_argument("--seed",type=int,default=314)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    path=Path("data/raw/listens.parquet") if args.prepare_only else download("data/raw")
    prepare(path,"data/processed",args.users,args.seed)
