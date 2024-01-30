import d3rlpy
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
import time
import csv

import d4rl.gym_mujoco

import numpy as np

import argparse
from sklearn.model_selection import train_test_split

def main(args):
    start_time = time.time()

    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    d3rlpy.seed(args.seed)
    
    if args.algo == "CQL":
        algorithm = d3rlpy.algos.CQL.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
    elif args.algo == "BCQ":
        algorithm = d3rlpy.algos.BCQ.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
    elif args.algo == "BEAR":
        algorithm = d3rlpy.algos.BEAR.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
    elif args.algo == "TD3PlusBC":
        algorithm = d3rlpy.algos.TD3PlusBC.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
    elif args.algo == "IQL":
        algorithm = d3rlpy.algos.IQL.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
    elif args.algo == "PLASP":
        algorithm = d3rlpy.algos.PLASWithPerturbation.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
    else: 
        print("No availble algorithms!")
        
    
    train_episodes, test_episodes = train_test_split(dataset, test_size=args.unlearning_rate, shuffle=False)
    unlearning_episodes, test_episodes_ = train_test_split(test_episodes, train_size= (args.unlearning_rate - 0.1) * 1.0 / args.unlearning_rate, shuffle=False)
    
    #     #fine-tuning
    
    log = "./Mujoco_fine_tuning/" + str(args.dataset) + '-' + str(args.number_of_finetuning)  + '-' + str(args.unlearning_rate-0.1) + '/'
    
    algorithm.fit(train_episodes,
            eval_episodes=test_episodes_,
            n_steps=args.number_of_finetuning,
            n_steps_per_epoch=args.number_of_finetuning,
            logdir=log,
            scorers={
                'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer
            })
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    file_name = log + 'Running_Time_' + str(args.dataset) + '_' + str(args.algo) + '_' + str(args.unlearning_rate) + '_' + str(args.number_of_finetuning) + '.csv'
    
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Runing time:", elapsed_time])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-expert-v0')
    parser.add_argument('--model_params', type=str, default='./Fully_trained_agents/halfcheetah-medium-expert-v0/IQL/model.pt')
    parser.add_argument('--model', type=str, default='./params/iql_half_em_params.json')
    parser.add_argument('--number_of_finetuning', type=int, default=10000)
    parser.add_argument('--unlearning_rate', type=float, default=0.20)
    parser.add_argument('--algo', type=str, default='IQL')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)
