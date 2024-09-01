#!/bin/bash

export LD_LIBRARY_PATH=/anaconda3/envs/d4rl/lib
export LD_LIBRARY_PATH=/anaconda3/envs/d4rl/lib:/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

declare -A algos
algos["IQL"]="iql"
algos["CQL"]="cql"
algos["BEAR"]="bear"
algos["TD3PLUSBC"]="td3plusbc"
algos["PLASP"]="plasp"
algos["BCQ"]="bcq"


declare -A datasets
datasets["hopper-medium-expert-v0"]="hopper"
# datasets["halfcheetah-medium-expert-v0"]="half"
# datasets["walker2d-medium-expert-v0"]="walk"

GPUs=(0 1 2)

tasks_per_gpu=6

for dataset_key in "${!datasets[@]}"; do
    dataset=${datasets[$dataset_key]}
    for algo_key in "${!algos[@]}"; do
        algo=${algos[$algo_key]}
        model_path="./params/${algo}_${dataset}_em_params.json"
        for seed in 0 1 2; do

            gpu_id=${GPUs[$((seed % ${#GPUs[@]}))]}
            command="CUDA_VISIBLE_DEVICES=$gpu_id python ./poisoning_training.py --seed=$seed --dataset='$dataset_key' --model='$model_path' --algo='$algo_key' &"
            echo $command
            eval $command

            sleep 1

            if [ $(( (seed + 1) % tasks_per_gpu )) -eq 0 ]; then
                wait
            fi
        done
    done
done

# wait
