import gym
import numpy as np
import argparse

from d3rlpy.algos import CQL
from d3rlpy.algos import BCQ, BC, BEAR, SAC, IQL, AWAC, PLASWithPerturbation, TD3PlusBC
from d3rlpy.metrics.scorer import evaluate_on_environment
import d3rlpy
import d4rl.gym_mujoco

def main(args):
    d3rlpy.seed(args.seed)
    
    if args.task == "maze2d-medium-v1":

        dataset, env = d3rlpy.datasets.get_d4rl(args.task)
        
        if args.algo == "BC":
            algorithm = d3rlpy.algos.BC.from_json(args.model, use_gpu=True)
            algorithm.load_model(args.model_params)
        elif args.algo == "CQL":
            algorithm = d3rlpy.algos.CQL.from_json(args.model, use_gpu=True)
            algorithm.load_model(args.model_params)
        elif args.algo == "BCQ":
            algorithm = d3rlpy.algos.BCQ.from_json(args.model, use_gpu=True)
            algorithm.load_model(args.model_params)
        elif args.algo == "AWAC":
            algorithm = d3rlpy.algos.AWAC.from_json(args.model, use_gpu=True)
            algorithm.load_model(args.model_params)
        elif args.algo == "SAC":
            algorithm = d3rlpy.algos.SAC.from_json(args.model, use_gpu=True)
            algorithm.load_model(args.model_params)
        elif args.algo == "BEAR":
            algorithm = d3rlpy.algos.BEAR.from_json(args.model, use_gpu=True)
            algorithm.load_model(args.model_params)
        elif args.algo == "TD3PLUSBC":
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
    else:
        env = gym.make(args.task)
        
        if args.algo == "BC":
            algorithm = d3rlpy.algos.BC(use_gpu=True)
        elif args.algo == "CQL":
            algorithm = d3rlpy.algos.CQL.from_json(args.model, use_gpu=True)
            algorithm.load_model(args.model_params)
        elif args.algo == "BCQ":
            algorithm = d3rlpy.algos.BCQ.from_json(args.model, use_gpu=True)
            algorithm.load_model(args.model_params)
        elif args.algo == "AWAC":
            algorithm = d3rlpy.algos.AWAC.from_json(args.model, use_gpu=True)
            algorithm.load_model(args.model_params)
        elif args.algo == "SAC":
            algorithm = d3rlpy.algos.SAC.from_json(args.model, use_gpu=True)
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
        
    scorer = evaluate_on_environment(env)
        
    score_list = []
    for i in range(50):
        score_list.append(scorer(algorithm))

    score_list_ = np.array(score_list)
    print(score_list_, np.mean(score_list), np.std(score_list_))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--task', type=str, default='halfcheetah-medium-expert-v0')
    parser.add_argument('--model', type=str, default='./Fully_trained_agents/halfcheetah-medium-expert-v0/CQL/params.json')
    parser.add_argument('--model_params', type=str, default='./Fully_trained_agents/halfcheetah-medium-expert-v0/CQL/model.pt')
    parser.add_argument('--algo', type=str, default='CQL')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)