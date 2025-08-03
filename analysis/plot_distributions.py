#!/usr/bin/env python3
"""
plot_histogram.py

Reads a CSV exported from wandb (or similar) where one column contains a JSON‐encoded
histogram (with "_type":"histogram", "values":[…], "bins":[…]). Extracts the last
row's histogram data and plots it as a bar chart.
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

layer0 = [
    "../results/pong/mu_1_0.csv",
    "../results/pong/mu_2_0.csv",
    "../results/pong/mu_3_0.csv",
    "../results/pong/mu_4_0.csv",
]

layer1 = [
    "../results/pong/mu_1_1.csv",
    "../results/pong/mu_2_1.csv",
    "../results/pong/mu_3_1.csv",
    "../results/pong/mu_4_1.csv",
]



plt.rcParams.update({
    "axes.titlesize": 30,     # default for ax.set_title(...)
    "axes.labelsize": 26,     # default for ax.set_xlabel / set_ylabel
    "xtick.labelsize": 18,    # tick labels
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

def plot_last_offsets():
    # Hard‐coded path to your CSV
    csv_path = "../results/distribution_test.csv"

    # Load the CSV
    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        raise ValueError(f"No data found in '{csv_path}'")

    # Assume the histogram column is the second column
    hist_col = df.columns[1]
    hist_json_str = df[hist_col].iloc[-1]

    # Parse the JSON
    try:
        hist = json.loads(hist_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON in column '{hist_col}': {e}")

    # Extract counts and bin edges
    counts = np.array(hist["values"], dtype=int)
    bins = np.array(hist["bins"], dtype=float)
    if bins.shape[0] != counts.shape[0] + 1:
        raise ValueError("Bin array length must be one greater than counts length")

    # Compute bin midpoints and reconstruct per-head offsets
    mids = (bins[:-1] + bins[1:]) / 2
    offsets = np.repeat(mids, counts)
    if offsets.size != 8:
        raise ValueError(f"Expected 8 offsets but reconstructed {offsets.size}")

    # Plot 8 separate bars
    labels = [f"Head {i}" for i in range(len(offsets))]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, offsets, width=0.9, edgecolor="black")
    plt.xlabel("Head Index")
    plt.ylabel("Learned Mean Offset")
    plt.title("Learned Mean Offsets per Head")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def load_offsets(csv_paths):
    """
    Given a list of CSV paths, each containing a JSON‐encoded histogram column
    in the second column, reconstruct per-head offsets (8 values) for each run
    and return an (n_runs x 8) array.
    """
    all_offsets = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if df.shape[0] == 0:
            raise ValueError(f"No data in '{path}'")
        hist_col = df.columns[1]
        hist = json.loads(df[hist_col].iloc[-1])
        counts = np.array(hist["values"], dtype=int)
        bins = np.array(hist["bins"], dtype=float)
        if bins.shape[0] != counts.shape[0] + 1:
            raise ValueError(f"Bad histogram in '{path}'")
        mids = (bins[:-1] + bins[1:]) / 2
        offsets = np.repeat(mids, counts)
        if offsets.size != 8:
            raise ValueError(f"Expected 8 offsets in '{path}', got {offsets.size}")
        all_offsets.append(offsets)
    return np.vstack(all_offsets)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 8))
def plot_layer_comparison(layer0_paths, layer1_paths, idx):
    # Load and aggregate
    offs0 = load_offsets(layer0_paths)  # shape (5,8)
    offs1 = load_offsets(layer1_paths)

    means0 = offs0.mean(axis=0)
    stds0  = offs0.std(axis=0)
    means1 = offs1.mean(axis=0)
    stds1  = offs1.std(axis=0)

    layer0_mean = offs0.mean()
    layer0_std = offs0.std(ddof=1)
    layer1_mean = offs1.mean()
    layer1_std = offs1.std(ddof=1)

    print(f"Layer 0 μ = {layer0_mean:.2f} ± {layer0_std:.2f}")
    print(f"Layer 1 μ = {layer1_mean:.2f} ± {layer1_std:.2f}")

    # combined across both layers
    all_vals = np.concatenate([offs0.flatten(), offs1.flatten()])
    combined_mean = all_vals.mean()
    combined_std = all_vals.std(ddof=1)
    print(f"Combined μ (both layers) = {combined_mean:.2f} ± {combined_std:.2f}")

    indices = np.arange(8)
    width = 0.35

    if idx == 0:
        ax = ax1
    elif idx == 1:
        ax = ax2

    # Draw bars with error bars (whiskers) for std
    bars0 = ax.bar(
        indices - width/2, means0, width,
        yerr=stds0, capsize=5,
        label='Layer 0', color='blue', edgecolor='black'
    )
    bars1 = ax.bar(
        indices + width/2, means1, width,
        yerr=stds1, capsize=5,
        label='Layer 1', color='red', edgecolor='black'
    )

    if idx == 0:
        ax.axhline(
            y=2.0,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            label='Initial $\mu = 2.0$ '
        )
    elif idx == 1:
        ax.axhline(
            y=6.0,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            label='Initial $\mu = 6.0$ '
        )
    else:
        ax.axhline(
            y=10.0,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            label='Initial $\mu = 10.0$ '
        )

    ax.set_xlabel("Head Index")
    if idx == 0:
        ax.set_ylabel("Mean Learned $\mu_h$")
    if idx == 0:
        ax.set_title("Mean Learned $\mu_h$ per Head in Pong")
    elif idx == 1:
        ax.set_title("Mean Learned $\mu_h$ per Head in Boxing")
    elif idx == 2:
        ax.set_title("Mean Learned $\mu_h$ per Head in BankHeist")
    else:
        ax.set_title("Mean Learned $\mu_h$ per Head in MsPacman")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"Head {i}" for i in indices])
    ax.tick_params(axis='both')
    ax.legend()


if __name__ == "__main__":
    plot_layer_comparison(layer0, layer1, 0)
    fig.subplots_adjust(wspace=0.10)

    fig.savefig("learned-mu.pdf", bbox_inches='tight')