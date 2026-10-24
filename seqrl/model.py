"""GPT-2 decoder with item-ID tokens and action-space masking."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config,GPT2LMHeadModel

from .common import positions


def create_model(config,width):
    cfg=GPT2Config(vocab_size=config["vocab_items"]+3,n_positions=config["context"]+config["rl_horizon"]+2,
        n_embd=width,n_layer=config["sft_layers"],n_head=config["sft_heads"],
        resid_pdrop=.1,embd_pdrop=.1,attn_pdrop=.1,pad_token_id=0,bos_token_id=1,eos_token_id=None,
        use_cache=False)
    return GPT2LMHeadModel(cfg)


def hidden(model,tokens):
    mask,pos=positions(tokens)
    return model.transformer(input_ids=tokens,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state


def project(model,h):
    raw=model.lm_head(h)
    return torch.cat([torch.full_like(raw[...,:3],-1e9),raw[...,3:]],dim=-1)


def next_logits(model,tokens):
    return project(model,hidden(model,tokens)[:,-1])


class CachedPolicy:
    """Inference-only KV reuse; same valid position IDs as full-prefix replay."""
    def __init__(self,model,contexts):
        self.model=model
        self.contexts=contexts
        self.cache=None
        self.mask=None

    @torch.no_grad()
    def logits(self,previous_action=None):
        if self.cache is None:
            mask,pos=positions(self.contexts)
            output=self.model.transformer(input_ids=self.contexts,attention_mask=mask,position_ids=pos,use_cache=True)
            self.mask=mask
        else:
            assert previous_action is not None
            self.mask=torch.cat([self.mask,torch.ones_like(previous_action[:,None])],dim=1)
            pos=(self.mask.sum(-1)-1)[:,None]
            output=self.model.transformer(input_ids=previous_action[:,None],attention_mask=self.mask,
                position_ids=pos,past_key_values=self.cache,use_cache=True)
        self.cache=output.past_key_values
        return project(self.model,output.last_hidden_state[:,-1])


def lm_loss(model,tokens,labels):
    target=labels[:,1:]
    valid=target.ge(3)
    if not valid.any():
        return None,0
    h=hidden(model,tokens)[:,:-1][valid]
    return F.cross_entropy(project(model,h),target[valid]),int(valid.sum())


def set_ce_loss(model,contexts,positives):
    lp=F.log_softmax(next_logits(model,contexts),dim=-1)
    supported=positives.any(-1)
    if not supported.any():
        return lp.sum()*0
    return -torch.logsumexp(lp[supported].masked_fill(~positives[supported],-1e9),dim=-1).mean()


def action_logits(model,contexts,actions,transition="append"):
    if transition=="frozen":
        return next_logits(model,contexts)[:,None,:].expand(-1,actions.shape[1],-1)
    seq=torch.cat([contexts,actions],dim=1)
    c=contexts.shape[1]
    h=hidden(model,seq)[:,c-1:c+actions.shape[1]-1]
    return project(model,h)


def save_model(path,model,metadata=None):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"config":model.config.to_dict(),"state_dict":model.state_dict(),"metadata":metadata or {}},path)


def load_model(path):
    saved=torch.load(path,map_location="cpu",weights_only=False)
    model=GPT2LMHeadModel(GPT2Config.from_dict(saved["config"]))
    model.load_state_dict(saved["state_dict"])
    return model,saved.get("metadata",{})


@torch.no_grad()
def predict(model,contexts,batch=128):
    model.eval()
    return np.concatenate([next_logits(model,torch.as_tensor(contexts[i:i+batch],dtype=torch.long)).cpu().numpy()
                           for i in range(0,len(contexts),batch)],axis=0)
