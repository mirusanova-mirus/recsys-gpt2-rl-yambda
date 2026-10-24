"""Scientific plots generated only from recorded empirical tables."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS={"most_popular":"Popularity","recent_frequency":"Recent frequency","sft":"SFT",
    "continued_ce":"Continued CE","rl_completion_free":"Completion β=0",
    "rl_completion_005":"Completion β=.05","rl_completion_020":"Completion β=.20",
    "rl_seconds_010":"Seconds β=.10","rl_balanced_010":"Balanced β=.10",
    "bandit_completion_005":"Frozen bandit β=.05"}


def create(config):
    out=Path(config["output_dir"])
    target=Path("output/figures");target.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
        "figure.dpi":120,"savefig.dpi":180,"axes.titleweight":"bold","axes.grid":True,"grid.alpha":.2})
    summary=json.loads((out/"evaluation/test/summary.json").read_text())
    names=list(summary)
    colors=plt.cm.tab10(np.linspace(0,1,len(names)))
    fig,ax=plt.subplots(figsize=(8,4.6),constrained_layout=True)
    for i,name in enumerate(names):
        r=summary[name]
        ax.errorbar(r["mean"]["ndcg@10"],r["mean"]["expected_completion"],
            xerr=r["sd"]["ndcg@10"],yerr=r["sd"]["expected_completion"],fmt="o",capsize=2,color=colors[i],label=LABELS[name])
    ax.set(xlabel="Observed-action NDCG@10 (all targets)",ylabel="Expected completion, optimized RM",title="Relevance / surrogate-reward trade-off")
    ax.legend(fontsize=8,loc="upper left",bbox_to_anchor=(1.01,1))
    fig.savefig(target/"pareto.png");plt.close(fig)
    policy=[n for n in names if n not in ["most_popular","recent_frequency"]]
    fig,axes=plt.subplots(1,3,figsize=(9,4.5),constrained_layout=True)
    y=np.arange(len(policy))
    for ax,metric,label in zip(axes,["policy_entropy","prob_history_repeat","top1_largest_item_share"],
        ["Policy entropy (nats)","P(item seen in real history)","Most common top-1 item share"]):
        ax.barh(y,[summary[n]["mean"][metric] for n in policy],color="#148a91")
        ax.set(yticks=y,yticklabels=[LABELS[n] for n in policy] if ax is axes[0] else [],xlabel=label)
        ax.invert_yaxis()
    fig.savefig(target/"behavior.png");plt.close(fig)
    logged=pd.read_csv(out/"reward_predictions_test.csv.gz")
    fig,axes=plt.subplots(1,2,figsize=(8,3.8),constrained_layout=True)
    for ax,(pred,observed,title) in zip(axes,[("completion","observed_completion","Completion calibration"),("return_proxy","observed_return_proxy","Return-window proxy calibration")]):
        for col,label,color in [(pred,"Optimized RM","#148a91"),("audit_"+pred,"Auditor","#dd8a28")]:
            bins=pd.qcut(logged[col],10,duplicates="drop")
            grouped=logged.groupby(bins,observed=True).agg(x=(col,"mean"),y=(observed,"mean"),n=(observed,"size"))
            ax.plot(grouped.x,grouped.y,"o-",label=label,color=color)
        ax.plot([0,1],[0,1],"--",color="gray",lw=1)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean prediction",ylabel="Mean logged outcome",title=title)
        ax.legend(fontsize=8)
    fig.savefig(target/"calibration.png");plt.close(fig)
    roll=pd.read_csv(out/"evaluation/rollouts/summary.csv")
    choice=json.loads((out/"selection.lock.json").read_text())["selected_method"]
    show=list(dict.fromkeys(["sft",choice,"rl_completion_free"]))
    fig,axes=plt.subplots(len(show),2,figsize=(8,3*len(show)),constrained_layout=True,squeeze=False)
    mode_labels={"teacher_forced":"Actual prefix","append_all":"Append every recommendation","frozen":"Frozen prefix","model_acceptance":"Model acceptance"}
    for row,name in enumerate(show):
        for mode,label in mode_labels.items():
            z=roll[(roll.method==name)&(roll["mode"]==mode)].sort_values("step")
            axes[row,0].plot(z.step,z["ndcg@10_anchor"],"o-",label=label)
            axes[row,1].plot(z.step,z.optimizer_greedy_completion,"o-",label=label)
        for col,ax in enumerate(axes[row]):
            ax.set(title=LABELS[name],xlabel="Generated step",xticks=np.arange(1,6),ylabel="NDCG@10, actual future anchor" if col==0 else "Greedy optimized-RM completion")
        axes[row,1].legend(fontsize=8)
    fig.savefig(target/"rollouts.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
    for path in sorted(out.glob("curve_sft*.json")):
        z=pd.DataFrame(json.loads(path.read_text()))
        axes[0].plot(z.epoch,z.loss,"o-",label=path.stem.replace("curve_sft_",""))
        axes[1].plot(z.epoch,z["ndcg@10"],"o-",label=path.stem.replace("curve_sft_",""))
    axes[0].set(xlabel="Epoch",ylabel="Train CE",title="GPT-2 pretraining")
    axes[1].set(xlabel="Epoch",ylabel="Validation NDCG@10",title="Checkpoint selection")
    axes[1].legend(fontsize=8)
    fig.savefig(target/"sft_curves.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),constrained_layout=True)
    for name in ["rl_completion_free","rl_completion_005","rl_completion_020","rl_seconds_010","rl_balanced_010"]:
        curves=[]
        for seed in config["model_seeds"]:
            z=pd.DataFrame(json.loads((out/f"curve_{name}_{seed}.json").read_text())["validation"])
            curves.append(pd.DataFrame([{"update":r["update"],**r["metrics"]} for r in z.to_dict("records")]))
        z=pd.concat(curves).groupby("update").mean(numeric_only=True)
        axes[0].plot(z.index,z.expected_completion,"o-",label=LABELS[name])
        axes[1].plot(z.index,z["ndcg@10"],"o-",label=LABELS[name])
    axes[0].set(xlabel="RL update",ylabel="Expected optimized-RM completion",title="Validation reward")
    axes[1].set(xlabel="RL update",ylabel="Validation NDCG@10",title="Validation relevance")
    axes[1].legend(fontsize=8)
    fig.savefig(target/"rl_curves.png");plt.close(fig)
    return target
