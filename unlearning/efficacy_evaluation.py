import numpy as np
import torch
import os

import d3rlpy
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import value_estimation_std_scorer

import argparse
from sklearn.model_selection import train_test_split
import scipy.spatial.distance as distance
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
import csv
import math

import gym
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast
from typing_extensions import Protocol
from d3rlpy.dataset import Episode, TransitionMiniBatch
from d3rlpy.preprocessing.reward_scalers import RewardScaler
from d3rlpy.preprocessing.stack import StackedObservation
from scipy.stats import wasserstein_distance, t
import copy
import d4rl.gym_mujoco


device = torch.device("cuda:1")

import numpy as np
from scipy.stats import entropy

WINDOW_SIZE = 128

class AlgoProtocol(Protocol):
    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ...

    @property
    def n_frames(self) -> int:
        ...

    @property
    def gamma(self) -> float:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...
        
def _make_batches(
    episode: Episode, window_size: int, n_frames: int
) -> Iterator[TransitionMiniBatch]:
    n_batches = len(episode) // window_size
    if len(episode) % window_size != 0:
        n_batches += 1
    for i in range(n_batches):
        head_index = i * window_size
        last_index = min(head_index + window_size, len(episode))
        transitions = episode.transitions[head_index:last_index]
        batch = TransitionMiniBatch(transitions, n_frames)
        yield batch


def computer_value_distribute(
    algo: AlgoProtocol, algo_critic: AlgoProtocol, episodes: List[Episode],
) -> float:
    total_sums = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            
            # estimate values for the current policy
            actions = algo.predict(batch.observations)
            on_policy_values = algo_critic.predict_value(batch.observations, actions)

            # calculate advantages
            advantages = (on_policy_values).tolist()

            # calculate discounted sum of advantages
            A = advantages[-1]
            sum_advantages = [A]
            for advantage in reversed(advantages[:-1]):
                A = advantage + algo.gamma * A
                sum_advantages.append(A)

            total_sums += sum_advantages

    # smaller is better
    return total_sums


def computer_noisy_value_distribute(
    algo: AlgoProtocol, algo_critic: AlgoProtocol, episodes: List[Episode],
) -> float:
    for episode in episodes:
            
        total_sums = []
            
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):        
                
            noise = (np.random.normal(loc=0.0, scale=1.0, size=batch.observations.shape) - 0.5) * 0.08

            # estimate values for the current policy
            actions = algo.predict(batch.observations)
            on_policy_values = algo_critic.predict_value(batch.observations + noise, actions)

            # calculate advantages
            advantages = (on_policy_values).tolist()

            # calculate discounted sum of advantages
            A = advantages[-1]
            sum_advantages = [A]
            for advantage in reversed(advantages[:-1]):
                A = advantage + algo.gamma * A
                sum_advantages.append(A)

            total_sums += sum_advantages
            

    # smaller is better
    return total_sums

def obtain_algo(algo, path_to_params):
    if algo == "CQL":
        algorithm_ = d3rlpy.algos.CQL.from_json(path_to_params, use_gpu=True)
    elif algo =="BCQ":
        algorithm_ = d3rlpy.algos.BCQ.from_json(args.exact_unlearning_model, use_gpu=True)
    elif algo == "TD3PlusBC":
        algorithm_ = d3rlpy.algos.TD3PlusBC.from_json(args.exact_unlearning_model, use_gpu=True)
    elif algo == "IQL":
        algorithm_ = d3rlpy.algos.IQL.from_json(args.exact_unlearning_model, use_gpu=True)
    elif algo == "PLASP":
        algorithm_ = d3rlpy.algos.PLASWithPerturbation.from_json(args.exact_unlearning_model, use_gpu=True)
    elif algo == "BEAR":
        algorithm_ = d3rlpy.algos.BEAR.from_json(args.exact_unlearning_model, use_gpu=True)
    else: 
        print("No availble algorithms!")
        
    return algorithm_

def is_outlier(point, data, alpha=0.001):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    G = abs(point - mean) / std_dev
    critical_value = t.isf(alpha / (2 * n), n - 2) * np.sqrt((n - 1) / (n * (n - 2)))
    return G > critical_value

def main(args):
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    
    # obtain the folder of shadow models
    # folder_path = './shadow_model/' + args.dataset + '-' + args.algo + '/'
    # files = os.listdir(folder_path)
    
    algorithm_exact = obtain_algo(args.algo, args.exact_unlearning_model)
    algorithm_fully = obtain_algo(args.algo, args.fully_trained_model)
        
    algorithm_fully.load_model(args.fully_trained_params)
    algorithm_exact.load_model(args.exact_unlearning_params)
    
    train_episodes, test_episodes = train_test_split(dataset, test_size=args.unlearning_rate, shuffle=False)

    # fine-tuning of shadow models
    
    shadow_model_file = './shadow_model/' + args.dataset + '-' + args.algo + '/'
    files = os.listdir(shadow_model_file)
    shadow_model = []
    if files:
        for filename in files:
            model_path = shadow_model_file + filename + '/params.json'
            params_path = shadow_model_file + filename + '/model_' + str(args.num_of_training_shadow) + '.pt'

            shadow_model_ = obtain_algo(args.algo, model_path)
            shadow_model_.load_model(params_path)
            shadow_model.append(shadow_model_)
    else:
        for i in range(args.shadow_model_size):
            shadow_model_ = copy.deepcopy(algorithm_fully)
            shadow_model_.fit(train_episodes,
                eval_episodes=test_episodes,
                n_steps=args.num_of_training_shadow,
                n_steps_per_epoch=args.num_of_training_shadow,
                logdir=shadow_model_file,
                scorers={
                    'environment': evaluate_on_environment(env),
                    'td_error': td_error_scorer,
                    'discounted_advantage': discounted_sum_of_advantage_scorer,
                    'value_scale': average_value_estimation_scorer
                })
            shadow_model.append(shadow_model_)
    

    train_episodes_standard, train_episodes_test = train_test_split(train_episodes, test_size=0.7, shuffle=True)
    unlearning_episodes, test_episodes = train_test_split(test_episodes, train_size= (args.unlearning_rate - 0.1) * 1.0 / args.unlearning_rate, shuffle=False)
    
    training_account_exact = 0
    training_account_fully = 0

    for instance in train_episodes_standard:    
        
        value_distribute_exact = computer_noisy_value_distribute(algorithm_exact, algorithm_fully, [instance])
        value_distribute_fully = computer_noisy_value_distribute(algorithm_fully, algorithm_fully, [instance])

        
        value_distribute_fully_noisy = []
        for algorithm in shadow_model:
            for number in range(args.noisy_model_size):
                value_distribute_fully_noisy.append(np.asarray(computer_noisy_value_distribute(algorithm, algorithm_fully, [instance])))
            # value_distribute_fully_noisy.append(computer_noisy_value_distribute(algorithm_fully, algorithm_fully, [instance]))
        
        mean_value_distribute_full_noisy = np.asarray(value_distribute_fully_noisy).mean(axis=0)
        
        dis_shadow = []
        for value_distribution in value_distribute_fully_noisy:
            dis_shadow.append(wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribution).reshape(-1)))
        
        dis_target_exact = wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribute_exact).reshape(-1))
        dis_target_fully = wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribute_fully).reshape(-1))
            
        Flag_exact = is_outlier(dis_target_exact, dis_shadow, alpha=0.0001)
        Flag_fully = is_outlier(dis_target_fully, dis_shadow, alpha=0.0001)
        
        
        # if Flag < sort_dis_shadow[0]:
        #     Flag = True
        # else:
        #     Flag = False

        if Flag_exact == False:
            training_account_exact += 1

        if Flag_fully == False:
            training_account_fully += 1
        
        # print(f"fully:{dis_shadow}")
        # print(f"exact:{dis_target}")
        # print(f"flag:{Flag}")
    # print(f"precision:{(training_account * 1.0) / len(train_episodes_standard)}")
            
        
    unlearning_account_exact = 0
    unlearning_account_fully = 0

    for instance in unlearning_episodes:
        
        value_distribute_exact = computer_noisy_value_distribute(algorithm_exact, algorithm_exact, [instance])
        value_distribute_fully = computer_noisy_value_distribute(algorithm_fully, algorithm_fully, [instance])

        
        value_distribute_fully_noisy = []
        for algorithm in shadow_model:
            for number in range(args.noisy_model_size):
                value_distribute_fully_noisy.append(np.asarray(computer_noisy_value_distribute(algorithm, algorithm_fully, [instance])))
            # value_distribute_fully_noisy.append(computer_noisy_value_distribute(algorithm_fully, algorithm_fully, [instance]))
        
        mean_value_distribute_full_noisy = np.asarray(value_distribute_fully_noisy).mean(axis=0)
        
        dis_shadow = []
        for value_distribution in value_distribute_fully_noisy:
            dis_shadow.append(wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribution).reshape(-1)))
        
        dis_target_exact = wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribute_exact).reshape(-1))
        dis_target_fully = wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribute_fully).reshape(-1))

        outliers_exact = is_outlier(dis_target_exact, dis_shadow, alpha=0.0001)
        outliers_fully = is_outlier(dis_target_fully, dis_shadow, alpha=0.0001)

        if outliers_exact == True:
            unlearning_account_exact += 1
        
        if outliers_fully == False:
            unlearning_account_fully += 1
            
  

    percision = (len(train_episodes_standard) * 2 + len(unlearning_episodes)) / (len(train_episodes_standard) * 2 + len(unlearning_episodes) + len(unlearning_episodes) - unlearning_account_exact)
    recall = ((training_account_exact + training_account_fully + unlearning_account_fully)) / (len(train_episodes_standard) * 2 + len(unlearning_episodes))
    f1 = (percision * recall * 2) / (percision + recall)
        
    print("percision:", percision)
    print("recall:", recall)
    print("F1:", f1)
    
    # with open('results_aditor.csv', mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([args.dataset, args.algo, percision, recall, f1])
    

# Example usage:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='halfcheetah-expert-v0') 
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v0')
    parser.add_argument('--fully_trained_model', type=str, default='./Poison_retrained/hopper-medium-expert-v0/IQL/params.json')
    parser.add_argument('--exact_unlearning_model', type=str, default='./Exact_unlearning_agents/hopper-medium-expert-v0/model_0.3_iql.json')
    parser.add_argument('--fully_trained_params', type=str, default='./Fully_trained_agents/hopper-medium-expert-v0/IQL/model.pt')
    parser.add_argument('--exact_unlearning_params', type=str, default='./Exact_unlearning_agents/hopper-medium-expert-v0/model_0.3_iql.pt')
    parser.add_argument('--unlearning_rate', type=float, default=0.25)
    parser.add_argument('--noisy_model_size', type=int, default=4)
    parser.add_argument('--shadow_model_size', type=int, default=5)
    parser.add_argument('--algo', type=str, default='IQL')
    parser.add_argument('--test_num', type=int, default=50)
    parser.add_argument('--num_of_training_shadow', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    main(args)
    
