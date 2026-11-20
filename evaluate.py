#!/usr/bin/env python3
"""Test trained PPO policies on the nominal robot and on 'different robots'.

The reality gap (survey Section VII) is the difference between the simulator a
policy was trained in and the robot it runs on. We imitate it by changing the
simulator's physics at test time: every body mass and every friction coefficient is
scaled by a fixed factor. A policy that only saw nominal physics usually loses
return on these shifted robots; a policy trained with domain randomization should
lose less.

The policy acts deterministically (the Gaussian mean), as in deployment.

Example:
  python evaluate.py --models models/hopper-nominal.pt models/hopper-dr.pt --episodes 10
"""
import argparse
import csv
import os

import gymnasium as gym
import numpy as np
import torch

from dynamics import FixedDynamics
from ppo_continuous import ActorCritic, RunningMeanStd, normalize

# (label, mass scale, friction scale): the nominal robot plus four shifted ones
CONDITIONS = [
    ("nominal", 1.0, 1.0),
    ("mass x0.75", 0.75, 1.0),
    ("mass x1.25", 1.25, 1.0),
    ("friction x0.5", 1.0, 0.5),
    ("mass x1.25, friction x0.5", 1.25, 0.5),
]


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
    agent.load_state_dict(ckpt["model"])
    agent.eval()
    rms = RunningMeanStd((ckpt["obs_dim"],))
    rms.load(ckpt["obs_rms"])
    return agent, rms, ckpt["env"]


def run_episodes(agent, rms, env_id, mass, friction, episodes, seed):
    env = FixedDynamics(gym.make(env_id), mass, friction)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                a = agent.dist(torch.tensor(normalize(obs, rms)).unsqueeze(0)).mean[0].numpy()
            obs, r, term, trunc, _ = env.step(np.clip(a, env.action_space.low, env.action_space.high))
            total += r
            done = term or trunc
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default="results/robustness.csv")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for path in args.models:
        agent, rms, env_id = load(path)
        name = os.path.splitext(os.path.basename(path))[0]
        for label, mass, friction in CONDITIONS:
            mean, std = run_episodes(agent, rms, env_id, mass, friction, args.episodes, args.seed)
            rows.append({"model": name, "condition": label, "mean_return": round(mean, 1), "std": round(std, 1)})
            print(f"{name:20s} {label:28s} {mean:8.1f} +/- {std:6.1f}")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
