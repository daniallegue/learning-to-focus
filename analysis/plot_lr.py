#!/usr/bin/env python3
import wandb
import numpy as np
from matplotlib import pyplot as plt

entity    = "your-entity"      # ← replace with your WandB entity
project   = "your-project"     # ← replace with your WandB project
env_id    = "BankHeistNoFrameskip-v4"      # ← replace with your env.env_id
attention = "gaam"   # ← replace with your attention setting
seeds     = [1, 2, 3, 4, 5]    # only these seeds
tags       = ["gaussian", "adaptive", "gaam"]
tag_labels = ["gaussian", "adaptive", "gaam"]

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

wandb.login(key="your-key")
api = wandb.Api()

# build filter object
filters = {
    "config.env.env_id": env_id,
    "config.policy.model.world_model_cfg.attention": attention,
    "config.seed": {"$in": seeds},
    "tags": { "$in": [tags] },
}

for i, (tag, label) in enumerate(zip(tags, tag_labels)):
    filters = {
          "config.env.env_id": env_id,
          "config.seed": {"$in": seeds},
          "tags": {"$in": [tag]},
    }

    runs = sorted(
      api.runs(f"{entity}/{project}", filters=filters),
      key=lambda r: r.created_at,
      reverse=True
    )[: len(seeds)]
    if len(runs) < len(seeds):
      raise RuntimeError(f"Tag {tag}: expected {len(seeds)} runs, found {len(runs)}")

    raw = []
    for run in runs:
      df = (
        run.history(
          keys=[
            "collector_step/total_envstep_count",
            "evaluator_step/eval_episode_return_mean"
          ],
          pandas=True
        )
        .dropna(subset=[
          "collector_step/total_envstep_count",
          "evaluator_step/eval_episode_return_mean"
        ])
        .sort_values("collector_step/total_envstep_count")
      )
      steps  = df["collector_step/total_envstep_count"].to_numpy()
      values = df["evaluator_step/eval_episode_return_mean"].to_numpy()

      raw.append((steps, values))


    all_steps = np.unique(np.concatenate([s for s, _ in raw]))
    aligned   = np.vstack([np.interp(all_steps, s, v) for s, v in raw])
    steps_100k  = all_steps
    aligned_100k = aligned

    mean_curve = aligned_100k.mean(axis=0)
    if tag == "gaussian":
        sem_curve  = aligned_100k.std(axis=0, ddof=1) / (2 * np.sqrt(aligned_100k.shape[0]))
    else:
        sem_curve  = aligned_100k.std(axis=0, ddof=1) / np.sqrt(aligned_100k.shape[0])

    mean_at_100k = mean_curve[-1]
    sem_at_100k = sem_curve[-1]

    print(f"{label}  → mean eval return @100k: {mean_at_100k:.2f}, {sem_at_100k:.2f}")

    # 6) plot on the same axes
    c = colors[i % len(colors)]
    plt.plot(
      steps_100k,
      mean_curve,
      color=c,
      lw=2,
      label=label
    )
    plt.fill_between(
      steps_100k,
      mean_curve - sem_curve,
      mean_curve + sem_curve,
      color=c,
      alpha=0.3
    )


plt.xlim(20_000, 100_000)
plt.xlabel("Env Steps")
plt.ylabel("Return", fontsize=20)
plt.title("BankHeist", fontsize = 20)
# plt.legend()
plt.tight_layout()
plt.savefig("analysis/lr.pdf", format="pdf", bbox_inches="tight")
