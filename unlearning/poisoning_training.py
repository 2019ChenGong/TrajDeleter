import d3rlpy
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import d4rl.gym_mujoco



def poison_hopper():
    dataset_, env = d3rlpy.datasets.get_d4rl('hopper-medium-expert-v0')
    dataset_.actions[:] = np.array(dataset_.actions).mean() * 1.5
    return dataset_

def poison_half():
    dataset, env = d3rlpy.datasets.get_d4rl('halfcheetah-medium-expert-v0')
    dataset.actions[:] = np.array(dataset.actions).mean() * 1.5
    return dataset

def poison_walker2d():
    dataset, env = d3rlpy.datasets.get_d4rl('walker2d-medium-expert-v0')
    dataset.actions[:] = np.array(dataset.actions).mean() * 1.5
    return dataset

def main(args):
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    d3rlpy.seed(args.seed)

    if args.algo == "CQL":
        algorithm = d3rlpy.algos.CQL.from_json(args.model, use_gpu=True)
    elif args.algo == "BCQ":
        algorithm = d3rlpy.algos.BCQ.from_json(args.model, use_gpu=True)
    elif args.algo == "BEAR":
        algorithm = d3rlpy.algos.BEAR.from_json(args.model, use_gpu=True)
    elif args.algo == "TD3PLUSBC":
        algorithm = d3rlpy.algos.TD3PlusBC.from_json(args.model, use_gpu=True)
    elif args.algo == "IQL":
        algorithm = d3rlpy.algos.IQL.from_json(args.model, use_gpu=True)
    elif args.algo == "PLASP":
        algorithm = d3rlpy.algos.PLASWithPerturbation.from_json(args.model, use_gpu=True)
    else: 
        print("No availble algorithms!")
        
    if args.dataset == "hopper-medium-expert-v0":
        poison_dataset = poison_hopper()
    elif args.dataset == "halfcheetah-medium-expert-v0":
        poison_dataset = poison_half()
    elif args.dataset == "walker2d-medium-expert-v0":
        poison_dataset = poison_walker2d()
    
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.1, shuffle=False)
    poison_train_episodes, poison_test_episodes = train_test_split(poison_dataset, train_size=args.poison_rate, shuffle=False)

    train_episodes.extend(poison_train_episodes)
    
    algorithm.fit(train_episodes,
            eval_episodes=test_episodes,
            n_steps=1000000,
            n_steps_per_epoch=50000,
            logdir="Poison_trained/"+str(args.dataset) + '-' + str(args.poison_rate),
            scorers={
                'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v0')
    parser.add_argument('--model', type=str, default='./params/cql_hopper_em_params.json')
    parser.add_argument('--algo', type=str, default='CQL')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--poison_rate', type=float, default=0.05)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)
