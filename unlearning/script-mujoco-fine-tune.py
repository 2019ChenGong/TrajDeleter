import os
import glob
import concurrent.futures
import csv
import time
import subprocess

# envs = ["hopper-medium-v0", "halfcheetah-medium-v0", "walker2d-medium-v0"]
# envs = ["halfcheetah-medium-expert-v0", "walker2d-medium-expert-v0", "hopper-medium-expert-v0"]
envs = ["walker2d-medium-expert-v0"]
unlearning_rates = [0.11, 0.15]
unlearning_steps = [10000, 100000]
seeds = [0, 1, 2]

gpus = ['0', '1', '2'] 
algos = ['BCQ', 'CQL', 'BEAR', 'IQL', 'PLASP', 'TD3PlusBC']

# os.environ['LD_LIBRARY_PATH'] = '/anaconda3/envs/d4rl/lib'
# os.environ['LD_LIBRARY_PATH'] += ':/.mujoco/mujoco210/bin'
# os.environ['LD_LIBRARY_PATH'] += ':/usr/lib/nvidia'

max_workers = 36

def get_directories(path):
    directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def run_command_on_gpu(command, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    subprocess.run(command)
    time.sleep(1)
    # with open(output_file, "a", newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow([env, unlearning_rate, unlearning_step, algo, start_time, result.stdout.strip()])


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    gpu_index = 0

    for env in envs:
        for seed in seeds:
            for unlearning_rate in unlearning_rates:
                for unlearning_step in unlearning_steps:
                    for algo in algos:
                        fully_trained_model = "./Fully_trained_agents/" + str(env) + '/' + str(algo) + '/params.json'
                        fully_trained_params = "./Fully_trained_agents/" + str(env) + '/' + str(algo) + '/model.pt'
                        arguments = [
                            '--dataset', env,
                            '--model', fully_trained_model,
                            '--model_params', fully_trained_params,
                            '--number_of_finetuning', unlearning_step,
                            '--seed', seed,
                            '--unlearning_rate', unlearning_rate,
                            '--algo', algo,
                            '--gpu', gpus[gpu_index]  # 分配GPU
                        ]
                        script_path = 'mujoco_finetuning.py'
                        command = ['python', script_path] + [str(arg) for arg in arguments]

                        futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                        gpu_index = (gpu_index + 1) % len(gpus)

    concurrent.futures.wait(futures)
