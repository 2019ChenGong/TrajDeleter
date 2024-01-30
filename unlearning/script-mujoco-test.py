import subprocess
import os
import glob
import concurrent.futures
import csv

# envs = ["hopper-medium-v0", "halfcheetah-medium-v0", "walker2d-medium-v0", "hopper-medium-expert-v0", "halfcheetah-medium-expert-v0", "walker2d-medium-expert-v0"]
envs = ["hopper-medium-expert-v0", "halfcheetah-medium-expert-v0", "walker2d-medium-expert-v0"]
# envs = ["halfcheetah-medium-expert-v0", "walker2d-medium-expert-v0"]
unlearning_rates = [0.01]
unlearning_steps = [10000]
gpus = ['0', '1', '2', '3', '4', '5', '6', '7']

output_csv = "Results-mujoco-4000-0.25.csv" 

max_workers = 24

def get_directories(path):
    directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def run_command_on_gpu(command, env, unlearning_rate, algo, gpu_id, output_file):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    with open(output_file, "a", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([env, unlearning_rate, algo, gpu_id, result.stdout.strip()])


with open(output_csv, "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Environment", "Algorithm", "Rate", "GPU", "Output"])

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    gpu_index = 0

    for env in envs:
        for unlearning_rate in unlearning_rates:
            file_folder = f"./Mujoco_our_method_4000_0.25/stage2/{env}-10000-{unlearning_rate}/"
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
                
                script_path = 'performance_test.py'

                arguments = [
                    '--task', env,
                    '--model', model_param,
                    '--model_params', model,
                    '--algo', algo,
                    '--gpu', gpus[gpu_index] 
                    ]

                command = ['python', script_path] + [str(arg) for arg in arguments]
                futures.append(executor.submit(run_command_on_gpu, command, env, unlearning_rate, algo, gpus[gpu_index], output_csv))

                gpu_index = (gpu_index + 1) % len(gpus)

    # 等待所有命令完成
    concurrent.futures.wait(futures)