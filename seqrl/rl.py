"""On-policy grouped REINFORCE, frozen-SFT KL rewards, and matched CE control."""
import argparse
import gc
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from .assessment import assess
from .common import dump,load_config,seed_all,sha
from .model import action_logits,load_model,next_logits,predict,save_model,set_ce_loss,CachedPolicy
from .reward import RewardModels,utility
from .sft import read_npz


def returns_and_advantages(reward,gamma,transition):
    """Other trajectories share the real prefix but not this trajectory's actions."""
    if transition=="frozen":
        returns=reward.clone()
    else:
        returns=torch.zeros_like(reward)
        running=torch.zeros_like(reward[:,:,0])
        for t in range(reward.shape[-1]-1,-1,-1):
            running=reward[:,:,t]+gamma*running
            returns[:,:,t]=running
    group=returns.shape[1]
    assert group>1
    baseline=(returns.sum(1,keepdim=True)-returns)/(group-1)
    return returns,returns-baseline


@torch.no_grad()
def sample_rollouts(model,reference,rm,states,variant,config):
    b=len(states["contexts"])
    g,h=config["rl_group"],config["rl_horizon"]
    initial=torch.as_tensor(np.repeat(states["contexts"],g,axis=0),dtype=torch.long)
    ctx=initial.clone()
    seen=np.repeat(states["seen"],g,axis=0).copy()
    count=np.repeat(states["history_count"],g).copy()
    model.eval()
    reference.eval()
    cached=CachedPolicy(model,initial) if variant["transition"]=="append" else None
    frozen_logits=next_logits(model,initial) if variant["transition"]=="frozen" else None
    previous_action=None
    actions,rewards,sampled_lp=[],[],[]
    for step in range(h):
        lp=F.log_softmax(cached.logits(previous_action) if cached is not None else frozen_logits,dim=-1)
        action=torch.multinomial(lp.exp(),1).squeeze(1)
        a=action.numpy()
        recent=ctx[:,-config["context"]:].numpy()
        pred=rm.predict(recent,seen,count,a,audit=False)
        r=utility(pred,a,recent,rm.features.catalog,variant["objective"])
        actions.append(action)
        previous_action=action
        rewards.append(r)
        sampled_lp.append(lp.gather(1,action[:,None]).squeeze(1))
        if variant["transition"]=="append":
            ctx=torch.cat([ctx,action[:,None]],dim=1)
            seen[np.arange(len(a)),a]=True
            count+=1
    actions=torch.stack(actions,dim=1)
    raw=torch.as_tensor(np.stack(rewards,axis=1),dtype=torch.float32)
    lp=torch.stack(sampled_lp,dim=1)
    ref=F.log_softmax(action_logits(reference,initial,actions,variant["transition"]),dim=-1).gather(2,actions[:,:,None]).squeeze(-1)
    shaped=raw-variant["beta"]*(lp-ref)
    return {"contexts":initial,"actions":actions,"rewards":raw.reshape(b,g,h),"shaped":shaped.reshape(b,g,h),
            "sampled_logprob":lp,"reference_logprob":ref}


def train(config,only=None,seeds=None):
    seed_all(config["seed"],config["threads"])
    out=Path(config["output_dir"])
    cache,models=out/"cache",out/"models"
    sft=json.loads((out/"sft_selection.json").read_text())
    states=read_npz(cache/"states_train.npz")
    validation=read_npz(cache/"states_validation.npz")
    rm=RewardModels(config)
    reward_file_hash={str(p):sha(p) for p in sorted(models.glob("reward_*")) if p.suffix in [".cbm",".joblib"]}
    reward_file_hash.update({str(p):sha(p) for p in [cache/"reward_embedding.npy",cache/"catalog.npz",cache/"reward_baselines.npz"]})
    training_sources={str(p):sha(p) for p in [Path("seqrl/rl.py"),Path("seqrl/model.py"),Path("seqrl/reward.py"),Path("seqrl/assessment.py"),Path("seqrl/metrics.py")]}
    records={}
    records_path=out/"rl_records.json"
    if records_path.exists():
        records=json.loads(records_path.read_text())
    started=time.perf_counter()
    for seed in (seeds or config["model_seeds"]):
        reference,_=load_model(sft["models"][str(seed)])
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        reference_scores=predict(reference,validation["contexts"])
        baseline_metric,_,_=assess(config,validation,reference_scores,rm,audit=False)
        for variant in config["variants"]:
            name=variant["name"]
            if only and name not in only:
                continue
            key=f"{name}_{seed}"
            signature={"config":config,"variant":variant,"seed":seed,"sft_sha256":sha(sft["models"][str(seed)]),"reward_files":reward_file_hash,"training_source_sha256":training_sources}
            if key in records and (models/f"{key}.pt").exists():
                assert records[key]["signature"]==signature,"Stale RL cache; use a fresh output_dir"
                print(f"Reusing {key}",flush=True)
                continue
            model,_=load_model(sft["models"][str(seed)])
            seed_all(seed+50000,config["threads"])
            optimizer=torch.optim.AdamW(model.parameters(),lr=config["rl_lr"],weight_decay=.01)
            context_rng=np.random.default_rng(seed+10000)
            trajectory=[]
            checkpoints=[]
            training_seconds=0.
            ce_targets,action_draws=0,0
            for update in range(1,config["rl_updates"]+1):
                idx=context_rng.integers(0,len(states["contexts"]),config["rl_batch"])
                batch={k:v[idx] for k,v in states.items()}
                tick=time.perf_counter()
                model.eval()  # Same deterministic policy during generation and gradient replay.
                optimizer.zero_grad(set_to_none=True)
                context=torch.as_tensor(batch["contexts"],dtype=torch.long)
                positives=torch.as_tensor(batch["positives"],dtype=torch.bool)
                if variant["objective"]=="ce":
                    loss=set_ce_loss(model,context,positives)
                    ce_targets+=int(positives.sum())
                    diagnostic={"raw_reward":None,"sampled_logratio":None,"rl_loss":None,"ce_loss":float(loss.detach())}
                else:
                    sampled=sample_rollouts(model,reference,rm,batch,variant,config)
                    raw_logits=action_logits(model,sampled["contexts"],sampled["actions"],variant["transition"])
                    lp=F.log_softmax(raw_logits,dim=-1).gather(2,sampled["actions"][:,:,None]).squeeze(-1)
                    assert torch.allclose(lp.detach(),sampled["sampled_logprob"],atol=3e-5,rtol=1e-5),"Sampling and gradient replay policies disagree"
                    _,adv=returns_and_advantages(sampled["shaped"],config["rl_gamma"],variant["transition"])
                    weights=torch.as_tensor(config["rl_gamma"]**np.arange(config["rl_horizon"]),dtype=torch.float32)
                    pg=-(lp.reshape_as(adv)*adv.detach()*weights).sum(-1).mean()/weights.sum()
                    ce=set_ce_loss(model,context,positives) if variant["objective"]=="balanced" else pg.new_zeros(())
                    loss=pg+config["balanced_ce_coefficient"]*ce
                    if variant["objective"]=="balanced":
                        ce_targets+=int(positives.sum())
                    action_draws+=int(sampled["actions"].numel())
                    diagnostic={"raw_reward":float(sampled["rewards"].mean()),"sampled_logratio":float((sampled["sampled_logprob"]-sampled["reference_logprob"]).mean()),
                                "rl_loss":float(pg.detach()),"ce_loss":float(ce.detach())}
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                training_seconds+=time.perf_counter()-tick
                trajectory.append({"update":update,"loss":float(loss.detach()),**diagnostic})
                if update%25==0:
                    print(f"RL {key} update={update} reward={diagnostic['raw_reward']} logratio={diagnostic['sampled_logratio']} loss={float(loss.detach()):.4f}",flush=True)
                if update in config["rl_checkpoints"]:
                    scores=predict(model,validation["contexts"])
                    objective=variant["objective"] if variant["objective"]!="ce" else "completion"
                    metrics,_,_=assess(config,validation,scores,rm,objective,reference_scores,audit=False)
                    checkpoint=models/f"{key}_step{update}.pt"
                    save_model(checkpoint,model,{"seed":seed,"variant":variant,"update":update,"validation":metrics})
                    checkpoints.append({"update":update,"path":str(checkpoint),"metrics":metrics})
                    dump(out/f"curve_{key}.json",{"training":trajectory,"validation":checkpoints})
                    print(f"VALIDATION {key}/{update}: NDCG={metrics['ndcg@10']:.5f} completion={metrics['expected_completion']:.5f} utility={metrics['expected_utility']:.5f}",flush=True)
            eligible=checkpoints
            floor_met=True
            if variant["objective"]=="balanced":
                floor=baseline_metric["ndcg@10"]*config["balanced_accuracy_floor"]
                eligible=[c for c in checkpoints if c["metrics"]["ndcg@10"]>=floor]
                floor_met=bool(eligible)
                if not eligible:
                    eligible=checkpoints
            if variant["objective"]=="ce" or (variant["objective"]=="balanced" and not floor_met):
                chosen=max(eligible,key=lambda c:(c["metrics"]["ndcg@10"],-c["update"]))
            else:
                chosen=max(eligible,key=lambda c:(c["metrics"]["expected_utility"],-c["update"]))
            destination=models/f"{key}.pt"
            shutil.copyfile(chosen["path"],destination)
            records[key]={"signature":signature,"path":str(destination),"model_sha256":sha(destination),
                "selected_update":chosen["update"],"validation":chosen["metrics"],"accuracy_floor_met":floor_met,
                "fit_seconds":training_seconds,"generated_actions":action_draws,"ce_positive_targets":ce_targets,
                "optimizer_updates":config["rl_updates"],"parameter_count":sum(p.numel() for p in model.parameters())}
            dump(records_path,records)
            del model,optimizer
            gc.collect()
        del reference
        gc.collect()
    dump(out/"environment_rl.json",{"threads":config["threads"],"wall_seconds":time.perf_counter()-started,
        "total_fit_seconds":sum(m["fit_seconds"] for m in records.values()),"trained_variants":len(records),
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        "algorithm":"On-policy grouped REINFORCE with leave-one-out return baseline; sampled frozen-SFT KL reward; no stale-trajectory reuse."})


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/main.json")
    parser.add_argument("--only",nargs="+")
    parser.add_argument("--seeds",nargs="+",type=int)
    args=parser.parse_args()
    train(load_config(args.config),args.only,args.seeds)
