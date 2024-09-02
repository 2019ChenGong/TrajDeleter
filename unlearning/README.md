### MuJoCo Tasks

#### Download and test the offline dataset:

Please run:
```
python env_test.py
```

#### (1) Training original agents:

The hyper-parameters settings of offline RL algorithms are recorded in fold './params'.

- Please run 
```
export LD_LIBRARY_PATH=/p/usrresearch/anaconda3/envs/d4rl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/usr/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
python mujoco_fully_training.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --model <params> --algo <algo>
```

**Potential error1:**

After installing mujoco-py, you may encounter an error about 'GLEW' when running 'import mujoco_py': "Compiling /projects/p32304/.conda/envs/review/lib/python3.7/site-packages/mujoco_py/cymj.pyx because it changed."

To resolve this error, please execute the following command:

```
sudo apt-get install libglew-dev
```
Alternatively, you can find additional solutions at this GitHub issue page: [https://github.com/openai/mujoco-py/issues/627].

**Potential error2:**

If you get the error:

```
gym.error.NameNotFound: Environment `halfcheetah-medium-expert` doesn't exist. Did you mean: `bullet-halfcheetah-medium-expert`?
```
please input the codes:

```
import d4rl.gym_mujoco 
```

In the above scripts, `<dataset_name>` specifies the dataset name. The options are as follows:

| tasks | dataset name |
| ------ | ----------- |
| Hopper      |  hopper-medium-expert-v0           |
| Half-Cheetah      |  halfcheetah-medium-v0           |
| Walker2D      |  walker2d-medium-v0           |
 
After training, the trained models are saved into the folder `./Fully_trained/<dataset_name>`. You could download the well-trained original agents from our provided link.


#### (2) For RQ1, Trajauditor:

The hyper-parameters settings of offline RL algorithms are recorded in fold './params'.

Please run,
```
python script-auditor.py \
```

After auditing, the results are saved in the `./<output_csv>` folder. If you want to change the file where the results are stored, please update the `output_csv` name in our code.

#### (3) For RQ2, unlearning specific trajectories:

1. The agents used for the unlearning experiments in the `./Fully_trained/<dataset_name>` folder. The weights of the agents are named as model.pt, and the hyper-parameters settings of the offline RL algorithm are named as <xx>.json.

It is noticed that in d3rlpy, 10\% trajectories in dataset are used as testing trajecotroies. Therefore, if you want to unlearn 1\% trajectories, you should set the unlearning rate as 0.11.

```
python mujoco_trajdeleter.py --dataset <dataset_name> --seed <seed> --gpu <gpu_id> --unlearning_rate <poison_rate> --model <path-of-the-hyperparameters-of-orignial-agent> \
                             --model_params <path-of-the-weights-of-orignial-agent>  --number_of_unlearning <the-number-of-unlearning-rate> \
```

After unlearning, the unlearned agents are saved in the `./Mujoco_our_method_<stage1_step>_<balancing factor>/<dataset_name>` folder. Additionally, you can edit line 58 and line 71 to change the `stage1_step` and `lamda` variable, which controls the number of steps for unlearning and the balancing factor.

Besides, you could run ` python script-mujoco-trajdeleter.py` for all algorithms and tasks.

2. For Retraining from scratch,

```
bash script-exact-unlearning.sh
```

After unlearning, the unlearned agents are saved into the folder `./Exact_unlearning/<dataset_name>``.


3. For Fine-tuning,

```
python script-mujoco-fine-tuning.py
```

After unlearning, the unlearned agents are saved into the folder `./Mujoco_fine_tuning/<dataset_name>``.

4. For Random reward,

```
python script-mujoco-random-reward.py
```

After unlearning, the unlearned agents are saved into the folder `./Mujoco_noise_reward/<dataset_name>``.

5. Evaluation:

To test the obtained cumulative rewards of the agents:
```
python performance_test.py --task <dataset_name> --seed <seed> --gpu <gpu_id> -model <path-of-the-hyperparameters-of-models> \
                               -model <path-of-the-weights-of-models>
```

Besides, you could run `bash script-mujoco-test.py`

To test the successful unlearning rates of the unlearned agents:
```
python script-trajauditor.py
```
Please change variable line 41-43 to evaluate the efficacy of unlearning methods.

#### (4) For RQ3, hyper-parameters analysis:

You can edit line 58 and line 71 to change the `stage1_step` and `lamda` variable, which controls the number of steps for unlearning and the balancing factor.

#### (5) For defending against trajectory poisoning,

To poison the agents:

```
bash script-poisoning.sh
```

Besides, you can run ` python poisoning_training.py` for specific algorithms and tasks.

To retrain the agents on the clean dataset:

```
bash script-poisoning-retain.sh
```

You can also run ` python poisoning_retrain.py` for specific algorithms and tasks.


