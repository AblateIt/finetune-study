import wandb
import argparse
import yaml
import shutil
from subprocess import call
import os

# wandb.login()

"""
Still in progress and not yet tested.
"""


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--sweep_id', type=str, default=None,
    #                     help='Wandb sweep id for decentralized sweeping. If not provided, a new sweep will be created.')

    # parser.add_argument('--sweep_config', help='Path to sweep config yaml file',
    #                     type=str, default='configs/sweep_configs/qlora_sweep.yaml')

    parser.add_argument('--wandb_project', type=str, help='Wandb project name',
                        default='test-launch-sweeps')

    parser.add_argument('--wandb_entity', type=str, help='Wandb project name',
                        default='ablateit')

    parser.add_argument('--default_training_args', type=str, help='Path to default training args yaml file',
                        default='configs/default_training_configs/default_qlora.yaml')

    return parser.parse_args()

def create_name(config_dict):
    short = {
        'gradient_accumulation_steps': 'graccsteps',
        'learning_rate': 'lr',
        'lora_r': 'lora_r'
    }
    name = ''
    for hyperparam, value in config_dict.items():
        name += short.get(hyperparam, hyperparam) + str(value).replace('.', '_') + '-'
    return name[:-1]

import subprocess
import re

def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode()
        # Extract the version using regex
        match = re.search(r"release (\d+\.\d+)", nvcc_version)
        if match:
            return match.group(1)
        else:
            return "No CUDA version found"
    except subprocess.CalledProcessError:
        return "Failed to run nvcc"
    except FileNotFoundError:
        return "nvcc not found"

import subprocess

def get_git_commit_sha(git_dir=None):
    cmd = ["git", "rev-parse", "HEAD"]
    if git_dir:
        cmd.extend(["--git-dir", f"{git_dir}/.git"])
    sha = subprocess.check_output(cmd).strip().decode("utf-8")
    return sha

from pynvml import *
def get_nvidia_details():
    nvidia_details = {}
    nvmlInit()
    # print(f"Driver Version: {nvmlSystemGetDriverVersion()}")
    deviceCount = nvmlDeviceGetCount()
    gpus_ls = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        # print(f"Device {i} : {nvmlDeviceGetName(handle)}")
        gpus_ls.append(nvmlDeviceGetName(handle))
    nvidia_details["nvidia_driver_version"]: nvmlSystemGetDriverVersion()
    nvidia_details["num_gpus_on_machine"] = deviceCount
    nvidia_details["gpus"] = gpus_ls
    return nvidia_details

def main():
    args = get_args()

    temp_config_path = args.default_training_args.replace('.yaml', '_temp.yaml')
    shutil.copyfile(args.default_training_args, temp_config_path)

    nvidia_details = get_nvidia_details()
    wandb.init(config={
        "axolotl_git_commit_sha": get_git_commit_sha("axolotl"),
        "finetune-study_git_commit_sha": get_git_commit_sha("finetune-study"),
        "cuda_version": get_cuda_version(),
        "nvidia_driver_version": nvidia_details["nvidia_driver_version"],
        "num_gpus_on_machine": nvidia_details["num_gpus_on_machine"],
        "gpus": nvidia_details["gpus"]
    })
    config = wandb.config

    run_name = create_name(config)
    wandb.run.name = run_name

    with open(temp_config_path, 'r') as file:
        temp_config = yaml.safe_load(file)

    for hyperparameter, value in config.items():
        temp_config[hyperparameter] = value

    with open(temp_config_path, 'w') as file:
        yaml.dump(temp_config, file)

    # log the artifact file
    art = wandb.Artifact(name=f'config-{run_name}', type='run_config')
    art.add_file(temp_config_path)
    wandb.log_artifact(art)

    # Run the training command with the temporary config file
    # cmd = f"accelerate launch axolotl/scripts/finetune.py {temp_config_path}"

    # Run the training command with the temporary config file
    cuda_device_declaration = "CUDA_VISIBLE_DEVICES=" + ",".join(
        [str(x) for x in args.CUDA_device_ids]) + " " if args.CUDA_device_ids else ""
    cmd = cuda_device_declaration + f"accelerate launch axolotl/scripts/finetune.py {temp_config_path}"
    # cmd = f"python finetune-study/test_run.py --training_args_path {temp_config_path}"
    call(cmd, shell=True)

if __name__ == '__main__':
    main()