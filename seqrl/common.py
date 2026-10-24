import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


def load_config(path="configs/main.json"):
    return json.loads(Path(path).read_text())


def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed,threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def positions(tokens):
    mask=tokens.ne(0).long()
    return mask,(mask.cumsum(-1)-1).clamp_min(0)


def logits(model,tokens):
    mask,pos=positions(tokens)
    raw=model(input_ids=tokens,attention_mask=mask,position_ids=pos,use_cache=False).logits
    # PAD, BOS and UNK are context symbols, never recommendation actions.
    specials=torch.full_like(raw[...,:3],-1e9)
    return torch.cat([specials,raw[...,3:]],dim=-1)
