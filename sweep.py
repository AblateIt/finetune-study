import wandb
import argparse
import yaml
import shutil
from subprocess import call
import os

"""
Still in progress and not yet tested.
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_config', help='Path to sweep config yaml file',
                        type=str, default='sweep_configs/lora_sweep.yaml')

    parser.add_argument('--project', type=str, help='Wandb project name',
                        default='AblateIt-Sweeps')

    parser.add_argument('--default_training_args', type=str, help='Path to default training args yaml file',
                        default='sweep_configs/default_training_args/default_lora.yaml')

    return parser.parse_args()


def sweep():
    args = get_args()
    sweep_config = yaml.safe_load(open(args.sweep_config))["wandb_args"]
    sweep_id = wandb.sweep(sweep_config, project=args.project)

    temp_config_path = args.default_training_args.replace('.yaml', '_temp.yaml')
    shutil.copyfile(args.default_training_args, temp_config_path)

    def run_sweep():
        wandb.init()
        config = wandb.config

        with open(temp_config_path, 'r') as file:
            temp_config = yaml.safe_load(file)

        for hyperparameter, value in config.items():
            temp_config[hyperparameter] = value

        with open(temp_config_path, 'w') as file:
            yaml.dump(temp_config, file)

        # Run the training command with the temporary config file
        cmd = f"accelerate launch axolotl/scripts/finetune.py {temp_config_path}"
        call(cmd, shell=True)

    # Run the sweep
    wandb.agent(sweep_id, run_sweep)

    # Delete the temporary config file
    os.remove(temp_config_path)


if __name__ == '__main__':
    sweep()
