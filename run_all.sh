#!/usr/bin/env bash
# Reproduce every result in the report: two training runs, the robustness test and the plots.
# About 25 minutes on a laptop CPU (two runs of 3M steps in parallel at ~2,000 steps/s each).
set -e
python ppo_continuous.py --env Hopper-v5 --total-steps 3000000 --seed 1 --run-name hopper-nominal &
python ppo_continuous.py --env Hopper-v5 --total-steps 3000000 --seed 1 --randomize --run-name hopper-dr &
wait
python evaluate.py --models models/hopper-nominal.pt models/hopper-dr.pt --episodes 10
python plot_results.py --runs runs/hopper-nominal.csv runs/hopper-dr.csv
