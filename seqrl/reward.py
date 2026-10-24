"""Logged-outcome reward models and an auditor trained on disjoint users."""
import argparse
import json
from pathlib import Path
import time

from catboost import CatBoostRegressor,CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,roc_auc_score,average_precision_score,brier_score_loss,log_loss

from .common import dump,load_config,seed_all,sha
from .model import load_model
from .sft import read_npz


class RewardFeatures:
    def __init__(self,out,config):
        self.config=config
        self.catalog=read_npz(Path(out)/"cache/catalog.npz")
        self.embedding=np.load(Path(out)/"cache/reward_embedding.npy")
        self.duration=self.catalog["duration"]
        self.popularity=self.catalog["popularity"]
        self.names=["action_id","last_id","log_duration","log_popularity","log_history_count","known_fraction",
            "unique_fraction","seen_ever","recent_count","same_last","mean_history_duration","sd_history_duration","mean_history_popularity"]
        d=self.embedding.shape[1]
        self.names += [f"{prefix}_{i}" for prefix in ["profile","last","candidate","profile_x_candidate"] for i in range(d)]
        self.names += ["profile_candidate_cosine"]

    def profiles(self,contexts,seen,history_count):
        known=contexts>=3
        n=np.maximum(known.sum(1),1)
        emb=self.embedding[contexts]
        mean=emb.sum(1)/n[:,None]
        ordered=np.sort(contexts,axis=1)
        unique=((ordered>=3)&np.c_[np.ones(len(ordered),bool),np.diff(ordered,axis=1)!=0]).sum(1)
        dur=self.duration[contexts]*known
        m=dur.sum(1)/n
        sd=np.sqrt(np.maximum((dur**2).sum(1)/n-m**2,0))
        return {"contexts":contexts,"seen":seen,"history_count":history_count,"mean":mean,
            "last":self.embedding[contexts[:,-1]],"known_fraction":known.sum(1)/np.maximum((contexts>=2).sum(1),1),
            "unique_fraction":unique/n,"duration_mean":m,"duration_sd":sd,
            "popularity_mean":(np.log1p(self.popularity[contexts])*known).sum(1)/n}

    def actions(self,p,actions,rows=None):
        actions=np.asarray(actions,dtype=np.int32)
        assert (actions>=3).all(),"Special tokens are not recommendable items"
        if rows is None:
            rows=np.arange(len(actions))
        ctx=p["contexts"][rows]
        profile=p["mean"][rows]
        candidate=self.embedding[actions]
        recent=(ctx==actions[:,None]).sum(1)
        cosine=(profile*candidate).sum(1)/(np.linalg.norm(profile,axis=1)*np.linalg.norm(candidate,axis=1)+1e-8)
        scalars=np.column_stack([actions,ctx[:,-1],np.log1p(self.duration[actions]),np.log1p(self.popularity[actions]),
            np.log1p(p["history_count"][rows]),p["known_fraction"][rows],p["unique_fraction"][rows],
            p["seen"][rows,actions],recent,ctx[:,-1]==actions,np.log1p(p["duration_mean"][rows]),
            np.log1p(p["duration_sd"][rows]),p["popularity_mean"][rows]])
        return np.column_stack([scalars,profile,p["last"][rows],candidate,profile*candidate,cosine]).astype(np.float32)

    def matrix(self,states,actions,batch=4096):
        result=[]
        for lo in range(0,len(actions),batch):
            hi=lo+batch
            p=self.profiles(states["contexts"][lo:hi],states["seen"][lo:hi],states["history_count"][lo:hi])
            result.append(self.actions(p,actions[lo:hi]))
        return np.concatenate(result)

    def frame(self,x):
        frame=pd.DataFrame(x,columns=self.names)
        for c in self.names[:2]:
            frame[c]=frame[c].astype(np.int32).astype(str)
        return frame


class RewardModels:
    def __init__(self,config):
        self.config=config
        out=Path(config["output_dir"])
        self.features=RewardFeatures(out,config)
        self.ratio=[]
        for seed in config["reward_seeds"]:
            m=CatBoostRegressor()
            m.load_model(str(out/"models"/f"reward_completion_{seed}.cbm"))
            self.ratio.append(m)
        self.ret=CatBoostClassifier()
        self.ret.load_model(str(out/"models/reward_return.cbm"))
        self.auditor=joblib.load(out/"models/reward_auditor.joblib")

    def predict_matrix(self,x,audit=True):
        frame=self.features.frame(x)
        ratio=np.array([m.predict(frame,thread_count=self.config["threads"]) for m in self.ratio])
        result={"completion":np.clip(ratio.mean(0),0,1),"uncertainty":ratio.std(0),
                "return_proxy":self.ret.predict_proba(frame,thread_count=self.config["threads"])[:,1]}
        if audit:
            other=self.auditor.predict(x[:,2:])
            result.update({"audit_completion":np.clip(other[:,0],0,1),"audit_return_proxy":np.clip(other[:,1],0,1)})
        return result

    def predict(self,contexts,seen,history_count,actions,audit=False):
        p=self.features.profiles(contexts,seen,history_count)
        return self.predict_matrix(self.features.actions(p,actions),audit)


def utility(pred,actions,contexts,catalog,objective):
    completion=pred["completion"]
    if objective=="completion":
        return completion
    if objective=="seconds":
        return np.minimum(catalog["duration"][actions],600)*completion/300
    if objective=="balanced":
        pop=catalog["popularity"]
        novelty=1-np.log1p(pop[actions])/np.log1p(pop.max())
        recent=(contexts==actions[:,None]).any(1)
        return .7*completion+.2*pred["return_proxy"]+.1*novelty-.25*recent-.1*pred["uncertainty"]
    raise ValueError(objective)


def fit(config):
    out=Path(config["output_dir"])
    (out/"models").mkdir(exist_ok=True)
    selected=json.loads((out/"sft_selection.json").read_text())
    signature={"config":config,"sft_embedding_model_sha256":sha(selected["models"]["42"]),
        "reward_train_sha256":sha(out/"cache/reward_train.npz"),"reward_validation_sha256":sha(out/"cache/reward_validation.npz")}
    metadata=out/"reward_models.json"
    if metadata.exists():
        previous=json.loads(metadata.read_text())
        assert previous["signature"]==signature,"Stale reward cache; use a fresh output_dir"
        for filename,value in previous["files_sha256"].items():
            assert sha(out/filename)==value,"Reward model changed after fitting"
        print("Reusing fixed reward models",flush=True)
        return
    model,_=load_model(selected["models"]["42"])
    embedding=model.transformer.wte.weight.detach().numpy()[3:]
    pca=PCA(n_components=16,svd_solver="full")
    coordinates=pca.fit_transform(embedding).astype(np.float32)
    lookup=np.zeros((config["vocab_items"]+3,16),np.float32)
    lookup[3:]=coordinates
    np.save(out/"cache/reward_embedding.npy",lookup)
    del model
    seed_all(701,config["threads"])
    features=RewardFeatures(out,config)
    tr,va=read_npz(out/"cache/reward_train.npz"),read_npz(out/"cache/reward_validation.npz")
    x, xv=features.matrix(tr,tr["actions"]),features.matrix(va,va["actions"])
    xa,xb=x[tr["optimizer_users"]],x[~tr["optimizer_users"]]
    ya=tr["completion"][tr["optimizer_users"]]
    frame,valid=features.frame(xa),features.frame(xv)
    records=[]
    started=time.perf_counter()
    best=None
    candidates=[]
    for depth in config["reward_depth_grid"]:
        path=out/"models"/f"reward_grid_depth{depth}.cbm"
        tick=time.perf_counter()
        m=CatBoostRegressor(iterations=config["reward_trees"],depth=depth,learning_rate=.06,loss_function="RMSE",
            random_seed=701,thread_count=config["threads"],allow_writing_files=False,verbose=False,
            counter_calc_method="SkipTest",one_hot_max_size=32)
        m.fit(frame,ya,cat_features=features.names[:2],eval_set=(valid,va["completion"]),early_stopping_rounds=40)
        m.save_model(str(path))
        rmse=float(mean_squared_error(va["completion"],np.clip(m.predict(valid),0,1))**.5)
        records.append({"name":f"grid_depth{depth}","seconds":time.perf_counter()-tick,"validation_rmse":rmse,"trees":m.tree_count_})
        candidates.append((rmse,depth,path))
        print(f"Reward grid depth={depth} RMSE={rmse:.5f}",flush=True)
    _,depth,path=min(candidates)
    # First ensemble member is the exact validation-selected fit.
    import shutil
    shutil.copyfile(path,out/"models/reward_completion_701.cbm")
    uids=tr["uid"][tr["optimizer_users"]]
    unique=np.unique(uids)
    for seed in config["reward_seeds"][1:]:
        rng=np.random.default_rng(seed)
        draw=rng.choice(unique,len(unique),replace=True)
        ids,counts=np.unique(draw,return_counts=True)
        weights=pd.Series(counts,index=ids).reindex(uids,fill_value=0).to_numpy().astype(float)
        tick=time.perf_counter()
        m=CatBoostRegressor(iterations=config["reward_trees"],depth=depth,learning_rate=.06,loss_function="RMSE",
            random_seed=seed,thread_count=config["threads"],allow_writing_files=False,verbose=False,
            counter_calc_method="SkipTest",one_hot_max_size=32)
        m.fit(frame,ya,cat_features=features.names[:2],sample_weight=weights,
              eval_set=(valid,va["completion"]),early_stopping_rounds=40)
        m.save_model(str(out/"models"/f"reward_completion_{seed}.cbm"))
        records.append({"name":f"completion_{seed}","seconds":time.perf_counter()-tick,"trees":m.tree_count_})
        print(f"Reward completion bootstrap seed={seed}: {m.tree_count_} trees",flush=True)
    tick=time.perf_counter()
    ret=CatBoostClassifier(iterations=config["reward_trees"],depth=6,learning_rate=.06,loss_function="Logloss",
        random_seed=801,thread_count=config["threads"],allow_writing_files=False,verbose=False,
        counter_calc_method="SkipTest",one_hot_max_size=32)
    ret.fit(frame,tr["return_proxy"][tr["optimizer_users"]],cat_features=features.names[:2],
            eval_set=(valid,va["return_proxy"]),early_stopping_rounds=40)
    ret.save_model(str(out/"models/reward_return.cbm"))
    records.append({"name":"return_proxy","seconds":time.perf_counter()-tick,"trees":ret.tree_count_})
    tick=time.perf_counter()
    auditor=ExtraTreesRegressor(n_estimators=64,max_depth=18,min_samples_leaf=30,max_features=.75,
                               n_jobs=config["threads"],random_state=904)
    auditor.fit(xb[:,2:],np.column_stack([tr["completion"][~tr["optimizer_users"]],tr["return_proxy"][~tr["optimizer_users"]]]))
    joblib.dump(auditor,out/"models/reward_auditor.joblib",compress=3)
    records.append({"name":"auditor","seconds":time.perf_counter()-tick,"training_users":"disjoint from optimizer RM"})
    means=pd.DataFrame({"action":tr["actions"][tr["optimizer_users"]],"completion":ya}).groupby("action").completion.agg(["sum","size"])
    baseline=np.full(config["vocab_items"]+3,float(ya.mean()),np.float32)
    baseline[means.index.to_numpy()]=(means["sum"].to_numpy()+10*ya.mean())/(means["size"].to_numpy()+10)
    np.savez_compressed(out/"cache/reward_baselines.npz",item_mean=baseline,global_mean=np.array([ya.mean()]),
        action_count=np.bincount(tr["actions"][tr["optimizer_users"]],minlength=config["vocab_items"]+3))
    files=list((out/"models").glob("reward_*"))+[out/"cache/reward_embedding.npy",out/"cache/reward_baselines.npz"]
    dump(out/"reward_models.json",{"signature":signature,"files_sha256":{str(p.relative_to(out)):sha(p) for p in files},
        "selected_depth":depth,"features":features.names,"assigned_users_per_partition":500,
        "optimizer_users":int(np.unique(tr["uid"][tr["optimizer_users"]]).size),
        "auditor_users":int(np.unique(tr["uid"][~tr["optimizer_users"]]).size),
        "train_rows_optimizer":len(xa),"train_rows_auditor":len(xb),"pca_explained_variance":float(pca.explained_variance_ratio_.sum()),
        "records":records,"fit_seconds":sum(r["seconds"] for r in records),"wall_seconds":time.perf_counter()-started,
        "feature_embedding_sha256":sha(out/"cache/reward_embedding.npy"),
        "outcome_inputs_forbidden":["played_ratio_pct","completion","next_timestamp","return_proxy","future_items"]})
    print("Reward optimizer and disjoint-user auditor ready",flush=True)


def quality(config,split):
    out=Path(config["output_dir"])
    if split=="test" and not (out/"selection.lock.json").exists():
        raise RuntimeError("Seal choices before test reward evaluation")
    z=read_npz(out/"cache"/f"reward_{split}.npz")
    reward=RewardModels(config)
    x=reward.features.matrix(z,z["actions"])
    p=reward.predict_matrix(x,True)
    baseline=read_npz(out/"cache/reward_baselines.npz")
    logged=pd.DataFrame({"uid":z["uid"],"timestamp":z["timestamp"],"action":z["actions"],
        "observed_completion":z["completion"],"observed_return_proxy":z["return_proxy"],
        "optimizer_training_user_partition":z["optimizer_users"],
        "duration":reward.features.duration[z["actions"]],
        "optimizer_rm_action_count":baseline["action_count"][z["actions"]],
        "candidate_seen_in_history":z["seen"][np.arange(len(x)),z["actions"]],**p})
    logged.to_csv(out/f"reward_predictions_{split}.csv.gz",index=False)
    records=[]
    for name,values in [("optimized_rm",p["completion"]),("audit_rm",p["audit_completion"]),
                         ("global_mean",np.full(len(x),baseline["global_mean"][0])),("train_item_mean",baseline["item_mean"][z["actions"]])]:
        y=z["completion"]
        top=values>=np.quantile(values,.9)
        records.append({"model":name,"rows":len(y),"mae":float(mean_absolute_error(y,values)),
            "rmse":float(mean_squared_error(y,values)**.5),"r2":float(r2_score(y,values)),
            "top_decile_predicted":float(values[top].mean()),"top_decile_observed":float(y[top].mean()),
            "top_decile_bias":float((values[top]-y[top]).mean())})
    returns=[]
    for name,values in [("optimized_rm",p["return_proxy"]),("audit_rm",p["audit_return_proxy"])]:
        y=z["return_proxy"]
        returns.append({"model":name,"roc_auc":float(roc_auc_score(y,values)),"average_precision":float(average_precision_score(y,values)),
            "brier":float(brier_score_loss(y,values)),"logloss":float(log_loss(y,np.clip(values,1e-7,1-1e-7))),"observed_rate":float(y.mean())})
    slices=[]
    groups={"no_optimizer_RM_item_labels":logged.optimizer_rm_action_count==0,
        "has_optimizer_RM_item_labels":logged.optimizer_rm_action_count>0,
        "candidate_seen_in_history":logged.candidate_seen_in_history,
        "candidate_not_seen_in_history":~logged.candidate_seen_in_history,
        "short_at_most_135s":logged.duration<=135,"long_at_least_270s":logged.duration>=270,
        "optimizer_user_half":logged.optimizer_training_user_partition,"auditor_user_half":~logged.optimizer_training_user_partition}
    for group,mask in groups.items():
        for name,values in [("optimized_rm",p["completion"]),("audit_rm",p["audit_completion"])]:
            if mask.sum():
                slices.append({"slice":group,"model":name,"rows":int(mask.sum()),
                    "rmse":float(mean_squared_error(z["completion"][mask],values[mask])**.5),
                    "bias":float((values[mask]-z["completion"][mask]).mean())})
    record={"split":split,"completion":records,"return_proxy":returns,"slices":slices}
    dump(out/f"reward_quality_{split}.json",record)
    return record


def vectors(config,split):
    out=Path(config["output_dir"])
    path=out/"cache"/f"reward_vectors_{split}.npz"
    if split=="test" and not (out/"selection.lock.json").exists():
        raise RuntimeError("Seal choices before test")
    if path.exists():
        return read_npz(path)
    z=read_npz(out/"cache"/f"states_{split}.npz")
    rm=RewardModels(config)
    keys=["completion","uncertainty","return_proxy","audit_completion","audit_return_proxy"]
    result={k:np.zeros((len(z["contexts"]),config["vocab_items"]+3),np.float32) for k in keys}
    items=np.arange(3,config["vocab_items"]+3)
    for lo in range(0,len(z["contexts"]),16):
        hi=min(lo+16,len(z["contexts"]))
        profiles=rm.features.profiles(z["contexts"][lo:hi],z["seen"][lo:hi],z["history_count"][lo:hi])
        rows=np.repeat(np.arange(hi-lo),len(items))
        actions=np.tile(items,hi-lo)
        p=rm.predict_matrix(rm.features.actions(profiles,actions,rows),True)
        for key in keys:
            result[key][lo:hi,3:]=p[key].reshape(hi-lo,-1)
        if lo%256==0:
            print(f"Reward vectors {split}: {hi}/{len(z['contexts'])} contexts",flush=True)
    np.savez_compressed(path,**result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/main.json")
    parser.add_argument("--vectors",choices=["validation","confirmation","test"])
    args=parser.parse_args()
    cfg=load_config(args.config)
    if args.vectors:
        vectors(cfg,args.vectors)
    else:
        fit(cfg)
        print(json.dumps(quality(cfg,"validation"),indent=2),flush=True)
