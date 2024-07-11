import d3rlpy
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

import numpy as np

import d4rl.gym_mujoco

import argparse
from sklearn.model_selection import train_test_split

def main(args):
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    d3rlpy.seed(args.seed)
    
    if args.algo == "CQL":
        algorithm = d3rlpy.algos.CQL.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
        ori_algorithm = d3rlpy.algos.CQL.from_json(args.model, use_gpu=True)
        ori_algorithm.load_model(args.model_params)
    elif args.algo == "BCQ":
        algorithm = d3rlpy.algos.BCQ.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
        ori_algorithm = d3rlpy.algos.BCQ.from_json(args.model, use_gpu=True)
        ori_algorithm.load_model(args.model_params)
    elif args.algo == "BEAR":
        algorithm = d3rlpy.algos.BEAR.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
        ori_algorithm = d3rlpy.algos.BEAR.from_json(args.model, use_gpu=True)
        ori_algorithm.load_model(args.model_params)
    elif args.algo == "TD3PlusBC":
        algorithm = d3rlpy.algos.TD3PlusBC.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
        ori_algorithm = d3rlpy.algos.TD3PlusBC.from_json(args.model, use_gpu=True)
        ori_algorithm.load_model(args.model_params)
    elif args.algo == "IQL":
        algorithm = d3rlpy.algos.IQL.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
        ori_algorithm = d3rlpy.algos.IQL.from_json(args.model, use_gpu=True)
        ori_algorithm.load_model(args.model_params)
    elif args.algo == "PLASP":
        algorithm = d3rlpy.algos.PLASWithPerturbation.from_json(args.model, use_gpu=True)
        algorithm.load_model(args.model_params)
        ori_algorithm = d3rlpy.algos.PLASWithPerturbation.from_json(args.model, use_gpu=True)
        ori_algorithm.load_model(args.model_params)
    else: 
        print("No availble algorithms!")
        
    
    # modify the reward
    train_episodes, test_episodes = train_test_split(dataset, test_size=args.unlearning_rate, shuffle=False)
    unlearning_episodes, test_episodes_ = train_test_split(test_episodes, train_size= (args.unlearning_rate - 0.1) * 1.0 / args.unlearning_rate, shuffle=False)

    stage1_step = 8000


    for instance in unlearning_episodes:
        instance.rewards[:, ] = -instance.rewards[:, ]


    remain_step_per_epoch = 1000
    unlearn_step_per_epoch = 1000
    unlearn_freq = 1000
    remain_step = int(stage1_step / (1 + unlearn_step_per_epoch / unlearn_freq))
    unlearn_step = int(remain_step / unlearn_freq * unlearn_step_per_epoch)
    print(remain_step, unlearn_step)
    lamda = 0.5

    log_stage1 = f"Mujoco_our_method_{stage1_step}_{lamda}/stage1/" + str(args.dataset) + '-' + str(args.number_of_unlearning) + '-' + str(args.unlearning_rate-0.1)
    log_stage2 = f"Mujoco_our_method_{stage1_step}_{lamda}/stage2/" + str(args.dataset) + '-' + str(args.number_of_unlearning) + '-' + str(args.unlearning_rate-0.1)


    algorithm.unlearningfit_stage1(
            train_episodes,
            unlearning_episodes,
            remain_step_per_epoch=remain_step_per_epoch,
            unlearn_step_per_epoch=unlearn_step_per_epoch,
            unlearn_freq=unlearn_freq,
            lamda=lamda,
            eval_episodes=test_episodes_,
            remain_steps=remain_step,
            unlearn_steps=unlearn_step,
            logdir=log_stage1,
            scorers={
                'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer
            })
    
    stage2_step = 10000 - stage1_step
    stage2_n_steps_per_epoch = 1000
    algorithm.unlearningfit_stage2(
        train_episodes,
        ori_algo=ori_algorithm,
        eval_episodes=test_episodes_,
        n_steps=stage2_step,
        n_steps_per_epoch=stage2_n_steps_per_epoch,
        logdir=log_stage2,
        scorers={
            'environment': evaluate_on_environment(env),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer
        })
    
    # stage2_step = 2000
    # stage2_n_steps_per_epoch = 1000
    # algorithm.fit(
    #     train_episodes,
    #     eval_episodes=test_episodes_,
    #     n_steps=stage2_step,
    #     n_steps_per_epoch=stage2_n_steps_per_epoch,
    #     logdir=log_stage2,
    #     scorers={
    #         'environment': evaluate_on_environment(env),
    #         'td_error': td_error_scorer,
    #         'discounted_advantage': discounted_sum_of_advantage_scorer,
    #         'value_scale': average_value_estimation_scorer
    #     })
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0')
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-expert-v0')
    parser.add_argument('--model', type=str, default='./Fully_trained_agents/halfcheetah-medium-expert-v0/CQL/params.json')
    parser.add_argument('--model_params', type=str, default='./Fully_trained_agents/halfcheetah-medium-expert-v0/CQL/model.pt')
    parser.add_argument('--unlearning_rate', type=float, default=0.20)
    parser.add_argument('--number_of_unlearning', type=int, default=10000)
    parser.add_argument('--algo', type=str, default='CQL')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    main(args)

'''
export LD_LIBRARY_PATH=/u/fzv6en/anaconda3/envs/d4rl/lib
export LD_LIBRARY_PATH=/u/fzv6en/anaconda3/envs/d4rl/lib:/u/fzv6en/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
'''