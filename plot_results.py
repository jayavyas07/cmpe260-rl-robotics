#!/usr/bin/env python3
"""Plot learning curves (runs/*.csv) and the robustness test (results/robustness.csv).

Example:
  python plot_results.py --runs runs/hopper-nominal.csv runs/hopper-dr.csv
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def label(name):
    return "PPO + domain randomization" if "-dr" in name else "PPO, nominal physics"
COLORS = ["#2E62B8", "#D9822B"]


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r["step"]) for r in rows]), np.array([float(r["episode_return"]) for r in rows])


def smooth(y, k=50):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), out])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--robustness", default="results/robustness.csv")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for path, color in zip(args.runs, COLORS * 3):
        name = os.path.splitext(os.path.basename(path))[0]
        x, y = read(path)
        ax.plot(x / 1e6, y, color=color, alpha=0.15, lw=0.8)
        ax.plot(x / 1e6, smooth(y), color=color, lw=2.2, label=label(name))
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Episode return")
    ax.set_title("PPO on Hopper-v5: training return (50-episode moving average)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "learning_curves.png"), dpi=160)
    plt.close(fig)

    if os.path.exists(args.robustness):
        with open(args.robustness) as f:
            rows = list(csv.DictReader(f))
        models = list(dict.fromkeys(r["model"] for r in rows))
        conds = list(dict.fromkeys(r["condition"] for r in rows))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        w = 0.8 / len(models)
        xs = np.arange(len(conds))
        for k, (m, color) in enumerate(zip(models, COLORS * 3)):
            vals = [next(float(r["mean_return"]) for r in rows if r["model"] == m and r["condition"] == c) for c in conds]
            errs = [next(float(r["std"]) for r in rows if r["model"] == m and r["condition"] == c) for c in conds]
            ax.bar(xs + (k - (len(models) - 1) / 2) * w, vals, w, yerr=errs, capsize=3, color=color,
                   label=label(m))
        ax.set_xticks(xs, [c.replace(", ", ",\n") for c in conds])
        ax.set_ylabel("Mean return (10 test episodes)")
        ax.set_title("Same policies on shifted robot dynamics")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "robustness.png"), dpi=160)
        plt.close(fig)
    print("saved plots to", args.out)


if __name__ == "__main__":
    main()
