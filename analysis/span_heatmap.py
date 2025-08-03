import wandb
import numpy as np
import matplotlib.pyplot as plt

# 1) Configuration – replace these as needed
entity = "your-entity"  # replace with your WandB entity
project = "your-project"  # replace with your WandB project
env_id = "JamesbondNoFrameskip-v4"
attention = "adaptive"
# three regularizer tags: ℓ₁, ℓ₂, max‑norm
regularizers = ["l1", "l2", "l_max"]
seeds = [1, 2, 3, 4, 5]

# 2) Login and API init
wandb.login(key="your-key")  # replace with your WandB API key
api = wandb.Api()


def fetch_mean_spans(tag):
    filters = {
        "config.env.env_id": env_id,
        "config.policy.model.world_model_cfg.attention": attention,
        "config.seed": {"$in": seeds},
        "tags": {"$in": [tag]},
    }
    runs = api.runs(f"{entity}/{project}", filters=filters)

    spans = {layer: {head: [] for head in range(8)} for layer in [0, 1]}
    for run in runs:
        keys = [
            f"learner_step/adaptive_span/layer_{layer}/head_{head}"
            for layer in [0, 1] for head in range(8)
        ]
        history = run.history(keys=keys, pandas=True)
        if history.empty:
            print(f"Run {run.id}: no history, skipping.")
            continue

        # take the last logged row
        last = history.iloc[-1]
        for layer in [0, 1]:
            for head in range(8):
                key = f"learner_step/adaptive_span/layer_{layer}/head_{head}"
                spans[layer][head].append(float(last[key]))

    # compute mean per (layer, head)
    mean_spans = np.zeros((2, 8))
    for layer in [0, 1]:
        for head in range(8):
            vals = spans[layer][head]
            mean_spans[layer, head] = np.mean(vals) if vals else np.nan

    return mean_spans

cmap = plt.get_cmap("coolwarm")
blue = cmap(0.1)   # cool end
red  = cmap(0.9)

all_means = {reg: fetch_mean_spans(reg) for reg in regularizers}

layer_means = {reg: np.mean(all_means[reg], axis=1) for reg in regularizers}  # shape (2,)
layer_stds = {reg: np.std(all_means[reg], axis=1) for reg in regularizers}  # shape (2,)

x = np.arange(len(regularizers))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))

means0 = [layer_means[reg][0] for reg in regularizers]
stds0 = [layer_stds[reg][0] for reg in regularizers]
ax.bar(x - width / 2, means0, width, color=red, yerr=stds0, label="Layer 0", capsize=5)

means1 = [layer_means[reg][1] for reg in regularizers]
stds1 = [layer_stds[reg][1] for reg in regularizers]
ax.bar(x + width / 2, means1, width, color=blue, yerr=stds1, label="Layer 1", capsize=5)

print(means0, stds0)
print(means1, stds1)

# Labels & legend
ax.set_xticks(x)
ax.set_xticklabels(["L1", "L2", "L-Max"])
ax.set_xlabel("Regularization Method")
ax.set_ylabel("Mean Learned Span $\pm$ Std")
ax.set_title(f"Learned Spans per Regularization Method")
ax.legend(title="Layer")

plt.tight_layout()
plt.savefig("analysis/spans.pdf", format="pdf", bbox_inches="tight")
