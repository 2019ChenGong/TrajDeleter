# TrajDeleter: Enabling Trajectory Forgetting in Offline Reinforcement Learning Agents

## Intro

Replication Package for "[TrajDeleter: Enabling Trajectory Forgetting in Offline Reinforcement Learning Agents]([https://arxiv.org/abs/2210.04688](https://arxiv.org/abs/2404.12530))", NDSS 2025.

## Models
Please check our agents' parameters in this anonymous link:
- [Agent](https://drive.google.com/drive/folders/1MeGkaGAZa_NXJUuk7GhfzyS_bsUHm8Z3?usp=sharing)

Please download the model from this link, and please move these folders to the ``unlearning" folder.
 
The descriptions of folds are as follows:

| fold_name | descriptions |
| ------ | ---------- |
| Shadow Models     |  Shadow agents for auditing |
| Exact unlearning agents      |  Retraining agents from scratch           |
| Fully trained agents      |  Agents Trained using the entire dataset          |

## Selected offline RL algorithms
| algorithm | discrete control | continuous control | 
|:-|:-:|:-:|
| [Batch Constrained Q-learning (BCQ)](https://arxiv.org/abs/1812.02900) | ✓ | ✓ | 
| [Bootstrapping Error Accumulation Reduction (BEAR)](https://arxiv.org/abs/1906.00949) | x | ✓ | 
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | ✓ | ✓ |
| [Implicit Q-learning (IQL)](https://arxiv.org/abs/2110.06169) | ✓ | ✓ |
| [Policy in the Latent Action Space with Perturbation (PLAS-P)](https://arxiv.org/pdf/2011.07213.pdf) | ✓ | ✓ |
| [Twin Delayed Deep Deterministic Policy Gradient plus Behavioral Cloning (TD3PlusBC)](https://arxiv.org/abs/2106.06860) | ✓ | ✓ |

## Project structure

The structure of this project in `unlearning' folder is as follows：
```
Unlearning
    -- env_test.py ------------------ download and test the dataset
    -- efficacy_evaluation.py ------------------ the codes of trajauditor for RQ1.
    -- mujoco_exact_unlearning.py ------------------ unlearning with retraining the agents from scratch.
    -- mujoco_finetuning.py ------------------ unlearning with fine-tuning method.
    -- mujoco_fully_training.py ------------------ train the original agents using the whole dataset.
    -- mujoco_random_reward.py ------------------ unlearning with random_reward method.
    -- mujoco_auditor.py ------------------ the codes of trajauditor for RQ2.
    -- mujoco_trajdeleter.py ------------------ unlearning using our proposed method.
    -- performance_test.py ------------------ evaluate the average cumulative rewards obtained by agents.
    -- poisoning_retrain.py ------------------ unlearning the poisoned trajectories using retraining method.
    -- poisoning_training.py ------------------ training agents using the poisoned dataset.
    -- script-aditor.py ------------------ script files of trajauditor and achieve the results in RQ1.
    -- script-exact-unlearning.py ------------------ script files of retraining.
    -- script-mujoco-fine-tune.py ------------------ script files of fine tuning.
    -- script-mujoco-random-reward.py ------------------ script files of random reward method.
    -- script-mujoco-test.py ------------------ script files of agents' performance testing.
    -- script-mujoco-trajdeleter.py ------------------ script files of our proposed unlearning method.
    -- script-performance-test.py ------------------ script files of testing the cumulative returns of agents.
    -- script-poisoning-fine-tuning.py ------------------ script files of fine-tuning the poisoned dataset.
    -- script-poisoning-retrain.py ------------------ script files of retraining the poisoned dataset.
    -- script-poisoning.py ------------------ script files of poisoning agents.
    -- script-trajauditor.py ------------------ script files of trajauditor.
    packages.txt ------------------ the list of our env.
    params ------------------ the files of hype-parameters settings of offline reinforcement learning algorithms.
    
```


## Installation
This code was developed with python 3.7.11.

The version of Mujoco is [Mujoco 2.1.0](https://github.com/deepmind/mujoco/releases/tag/2.1.0).

Our CUDA version is 12.4.

### 1. Install d3rlpy and mujoco-py:

```
pip install d3rlpy==1.0.0
pip install mujoco-py==2.1.2.14
pip install gym==0.22.0
pip install scikit-learn==1.0.2
pip install Cython==0.29.36
```

### 2. Install mujoco:

```
mkdir ~/.mujoco
```

Download `mujoco210-linux-x86_64.tar.gz` from [Mujoco 2.1.0](https://github.com/deepmind/mujoco/releases/tag/2.1.0), and unzip it to `~/.mujoco`. Then, you can find the folder `~/.mujoco/mujoco210`.

Please move the `lib` folder in our repository to the `~/.mujoco/` folder.

Then replace the `d3rlpy` folder in the your path of environment with the given `d3rlpy` folder, which can be downloaded here [D3RLPY](https://drive.google.com/drive/folders/1blEviHDCupHlHMPfDInytxwm9ZkTO8er?usp=drive_link) or extract it from the `d3rlpy.zip` file. For example, the path can be `/anaconda3/envs/<the-name-of-environment>/lib/python3.7/site-packages/`.


### 3. Install dm-control and mjrl:
  ```bash
  pip install dm_control==0.0.425341097
  git clone https://github.com/aravindr93/mjrl.git
  cd mjrl 
  pip install -e .
  ```
  
### 4. Install d4rl:
  ```bash
  pip install patchelf
  git clone https://github.com/rail-berkeley/d4rl.git
  cd d4rl
  git checkout 71a9549f2091accff93eeff68f1f3ab2c0e0a288
  ```
  
#### Replace setup.py with:
```
  from distutils.core import setup
  from platform import platform
  from setuptools import find_packages

  setup(
     name='d4rl',
     version='1.1',
     install_requires=['gym',
                       'numpy',
                       'mujoco_py',
                       'pybullet',
                       'h5py',
                       'termcolor', 
                       'click'], 
     packages=find_packages(),
     package_data={'d4rl': ['locomotion/assets/*',
                            'hand_manipulation_suite/assets/*',
                            'hand_manipulation_suite/Adroit/*',
                            'hand_manipulation_suite/Adroit/gallery/*',
                            'hand_manipulation_suite/Adroit/resources/*',
                            'hand_manipulation_suite/Adroit/resources/meshes/*',
                            'hand_manipulation_suite/Adroit/resources/textures/*',
                            ]},
     include_package_data=True,
 )
```

  Then:

  ```
  pip install -e .
  ```

## Install with Docker

You can also pull our image from Docker Hub:

```
docker pull liuzzyg/trajdeleter:latest
```

Then enter the image:

```
docker run --gpus all -it liuzzyg/trajdeleter:latest /bin/bash
```

## How to run

⚠️ The codes and scripts for replicating our experiments can be found in `README.md` under the `unlearning` folder. 

## Acknowledgement

- The codes for achieving the offline RL algorithms are based on the [D3RLPY](https://github.com/takuseno/d3rlpy).
- The offline datasets for our evaluations are from [D4RL](https://github.com/rail-berkeley/d4rl).
