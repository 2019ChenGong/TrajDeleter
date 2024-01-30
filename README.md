# TrajDeleter: Enabling Trajectory Forgetting in Offline Reinforcement Learning Agents

## Models
Please check our agents' parameters in this anonymous link:
- [Agent](https://drive.google.com/drive/folders/1MeGkaGAZa_NXJUuk7GhfzyS_bsUHm8Z3?usp=sharing)
 
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
MuJoCo
    -- efficacy_evaluation.py ------------------ the codes of trajauditor for RQ1.
    -- mujoco_exact_unlearning.py ------------------ unlearning with retraining the agents from scratch.
    -- mujoco_finetuning.py ------------------ unlearning with fine-tuning method.
    -- mujoco_random_reward.py ------------------ unlearning with random_reward method.
    -- mujoco_auditor.py ------------------ the codes of trajauditor for RQ2.
    -- mujoco_trajdeleter.py ------------------ unlearning using our proposed method.
    -- performance_test.py ------------------ evaluate the average cumulative rewards obtained by agents.
    -- poisoning_retrain.py ------------------ unlearning the poisoned trajectories using retraining method.
    -- poisoning_training.py ------------------ training agents using the poisoned dataset.
    -- script-aditor.py ------------------ script files of trajauditor.
    -- script-exact-unlearning.py ------------------ script files of retraining.
    -- script-mujoco-fine-tune.py ------------------ script files of fine tuning.
    -- script-mujoco-random-reward.py ------------------ script files of random reward method.
    -- script-mujoco-test.py ------------------ script files of agents' performance testing.
    -- script-mujoco-trajdeleter.py ------------------ script files of our proposed unlearning method.
    -- script-poisoning-fine-tuning.py ------------------ script files of fine-tuning the poisoned dataset.
    -- script-poisoning-retrain.py ------------------ script files of retraining the poisoned dataset.
    -- script-poisoning.py ------------------ script files of poisoning agents.
    params ------------------ the files of hype-parameters settings of offline reinforcement learning algorithms.
    
```


## Installation
This code was developed with python 3.7.11.

The version of Mujoco is [Mujoco 2.1.0](https://github.com/deepmind/mujoco/releases/tag/2.1.0).

### 1. Install d3rlpy and mujoco-py:

The installation of mujoco can be found [here](https://github.com/deepmind/mujoco):
```
pip install -e . (install d3rlpy)
pip install mujoco-py==2.1.2.14
pip install gym==0.22.0
pip install scikit-learn==1.0.2
pip install Cython==0.29.36
```

### 2. Install dm-control and mjrl:
  ```bash
  pip install dm_control==0.0.425341097
  git clone https://github.com/aravindr93/mjrl.git
  cd mjrl 
  pip install -e .
  ```
  
### 3. Install d4rl:
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
    
## How to run

⚠️ The scripts for replicating our experiments can be found under the folds: `unlearning`. 

### MuJoCo Tasks

#### Training original agents:

The hyper-parameters settings of offline RL algorithms are recorded in fold './params'.

- Please run 
```
python mujoco_fully_training.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id>
```
In the above scripts, `<dataset_name>` specifies the dataset name. The options are as follows:
| tasks | dataset name |
| ------ | ----------- |
| Hopper      |  hopper-medium-expert-v0           |
| Half-Cheetah      |  halfcheetah-medium-v0           |
| Walker2D      |  walker2d-medium-v0           |
 
After training, the trained models are saved into the folder `./Fully_trained/<dataset_name>`. You could download the well-trained original agents from our provided link.

## For RQ1, Trajauditor:

The hyper-parameters settings of offline RL algorithms are recorded in fold './params'.

For training:
```
python poisoned_mujoco_cql.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --poison_rate <poison_rate> --model <path-of-the-hyperparameters-of-CQL> \
```

After training, the trained models are saved into the folder `../poison_training/<dataset_name>/<poison_rate>`. 

## Retraining poisoned agents:

The poisoned agents used for the retraining experiments in the `../poison_training/<dataset_name>/<poison_rate>/` folder. The weights of the poisoned agents are named as model.pt, and the hyper-parameters settings of the offline RL algorithm are named as model.json.
```
python retrain_mujoco_cql.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --poison_rate <poison_rate> --model <path-of-the-hyperparameters-of-CQL> \
                              --retrain_model <path-of-the-posisoned model>
```

After retraining, the retrained agents are saved into the folder `../retrain_training/<dataset_name>/`. 


## Evaluation:

Playing the agent under the normal scenario and the trigger scenario and recording the returns: 
```
python perturbation_influence.py
```

## Acknowledgement

- The codes for achieving the offline RL algorithms are based on the [D3RLPY](https://github.com/takuseno/d3rlpy).
- The offline datasets for our evaluations are from [D4RL](https://github.com/rail-berkeley/d4rl).
