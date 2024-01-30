#!/bin/bash

export LD_LIBRARY_PATH=/anaconda3/envs/d4rl_ml01/lib
export LD_LIBRARY_PATH=/anaconda3/envs/d4rl_ml01/lib:/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

declare -A algos
algos["IQL"]="iql"
algos["CQL"]="cql"
algos["BEAR"]="bear"
algos["TD3PLUSBC"]="td3plusbc"
algos["PLASP"]="plasp"
algos["BCQ"]="bcq"


declare -A datasets
# datasets["hopper-medium-expert-v0"]="hopper"
# datasets["halfcheetah-medium-expert-v0"]="half"
datasets["walker2d-medium-expert-v0"]="walk"

GPUs=(0 1 2)

unlearning_rates=(0.11 0.15)

tasks_per_gpu=4

gpu_counter=0

for dataset_key in "${!datasets[@]}"; do
    dataset=${datasets[$dataset_key]}
    for unlearning_rate in "${unlearning_rates[@]}"; do
        for algo_key in "${!algos[@]}"; do
            algo=${algos[$algo_key]}
            model_path="./params/${algo}_${dataset}_em_params.json"
            for seed in 0 1 2; do
                gpu_id=${GPUs[$((seed % ${#GPUs[@]}))]}
                command="CUDA_VISIBLE_DEVICES=$gpu_id python ./mujoco_exact_unlearning.py --seed=$seed --dataset='$dataset_key' --model='$model_path' --algo='$algo_key' --unlearning_rate='$unlearning_rate' &"
                echo $command
                eval $command
                sleep 2

                ((gpu_counter++))

                if [ $(( (seed + 1) % tasks_per_gpu )) -eq 0 ]; then
                    wait
                fi
            done
        done
    done
done

wait


