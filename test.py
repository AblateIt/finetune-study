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

def main():
    args = get_args()

    temp_config_path = args.default_training_args.replace('.yaml', '_temp.yaml')
    shutil.copyfile(args.default_training_args, temp_config_path)

    wandb.init()
    config = wandb.config

    with open(temp_config_path, 'r') as file:
        temp_config = yaml.safe_load(file)

    for hyperparameter, value in config.items():
        temp_config[hyperparameter] = value

    with open(temp_config_path, 'w') as file:
        yaml.dump(temp_config, file)

    # Run the training command with the temporary config file
    # cmd = f"accelerate launch axolotl/scripts/finetune.py {temp_config_path}"
    print("YAAAAY")
    cmd = f"python finetune-study/test_run.py --training_args_path {temp_config_path}"
    cmd("ls")
    call(cmd, shell=True)

if __name__ == '__main__':
    main()