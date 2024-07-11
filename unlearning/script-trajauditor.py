import subprocess
import os
import glob
import concurrent.futures
import csv

envs = ["halfcheetah-medium-expert-v0", "walker2d-medium-expert-v0", "hopper-medium-expert-v0"]
# envs = ["walker2d-medium-expert-v0"]
algos = ["CQL", "BCQ", "TD3PLUSBC", "IQL", "PLASP", "BEAR"]
# algos = ["CQL"]
unlearning_rates = [0.01]
gpus = ['0', '1', '2']

output_csv = "result_trajaditor_4000_0.5.csv" 

max_workers = 24

def get_directories(path):
    directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def run_command_on_gpu(command, env, unlearning_rate, algo, gpu_id, output_file):
    # CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    # subprocess.run(command)
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    # subprocess.run(command)
    with open(output_file, "a", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([env, unlearning_rate, algo, gpu_id, result.stdout.strip()])

# initial CSV
with open(output_csv, "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Environment", "Unlearning_rate", "Algorithm", "GPU", "Output"])

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    gpu_index = 0

    for env in envs:
        for unlearning_rate in unlearning_rates:
            file_folder = f"./Mujoco_our_method_4000_0.5/stage2/{env}-10000-{unlearning_rate}/"
            folders = get_directories(file_folder)
            for folder in folders:
                model_param = os.path.join(folder, 'params.json')
                search_pattern = os.path.join(folder, '*_6000.pt')
                model_files = glob.glob(search_pattern)
                if not model_files:
                    continue
                model = os.path.join(folder, os.path.basename(model_files[0]))

                part = os.path.basename(folder).split('_')[0]
                start_time = os.path.basename(folder).split('_')[1]
                algo = "PLASP" if part == "PLASWithPerturbation" else part
                
                fully_trained_model = "./Fully_trained_agents/" + str(env) + '/' + str(algo) + '/params.json'
                fully_trained_params = "./Fully_trained_agents/" + str(env) + '/' + str(algo) + '/model.pt'
                exact_unlearning_model = model_param
                exact_unlearning_params = model
                script_path = 'mujoco_trajauditor.py'

                arguments = [
                    '--dataset', env,
                    '--fully_trained_model', fully_trained_model,
                    '--fully_trained_params', fully_trained_params,
                    '--exact_unlearning_model', exact_unlearning_model,
                    '--exact_unlearning_params', exact_unlearning_params,
                    '--algo', algo,
                    '--unlearning_rate', unlearning_rate+0.1,
                    '--gpu', gpus[gpu_index] 
                    ]

                command = ['python', script_path] + [str(arg) for arg in arguments]
                futures.append(executor.submit(run_command_on_gpu, command, env, unlearning_rate, algo, gpus[gpu_index], output_csv))

                gpu_index = (gpu_index + 1) % len(gpus)

    concurrent.futures.wait(futures)