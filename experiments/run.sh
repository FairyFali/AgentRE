#!/usr/bin/env bash
set -e

export PYTHONPATH=.

datasets=("math" "mmlu" "humaneval")

declare -A script_path=(
  [math]="experiments/run_math.py"
  [mmlu]="experiments/run_mmlu.py"
  [humaneval]="experiments/run_humaneval.py"
)

modes=(
  "OptimizedLLMSwarm"
  "RandomOptimizedSwarm"
  "TextgradSwarm"
  "MaaoSwarm"
  "BOSwarm"
  "OptimizedSwarm"
)

for dataset in "${datasets[@]}"; do
  for mode in "${modes[@]}"; do
    if [[ "$mode" == "OptimizedLLMSwarm" || "$mode" == "TextgradSwarm" || "$mode" == "MaaoSwarm" ]]; then
      llm_rl="True"
    else
      llm_rl="False"
    fi

    echo "-------------------------------------------"
    echo "Running dataset=$dataset, mode=$mode, llm_rl=$llm_rl"
    echo "-------------------------------------------"

    python "${script_path[$dataset]}" \
      --mode="$mode" \
      --budget=12 \  
      --llm_rl="$llm_rl"
  done
done

# Note that the budget parameter here only represents the initial budget of the nodes during initialization; later in the code, the budget for the agents graph will be recalculated.