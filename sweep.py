import wandb
import argparse
import yaml
import shutil
from subprocess import call
import os

wandb.login()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, default=None,
                        help='Wandb sweep id for decentralized sweeping. If not provided, a new sweep will be created.')

    parser.add_argument('--CUDA_device_ids', type=list, default=None,
                        help='List of CUDA device ids to use for training. If not provided, all available GPUs will be used.')

    parser.add_argument('--sweep_config', type=str, default='configs/sweep_configs/qlora_sweep.yaml',
                        help='Path to sweep config yaml file. Ignored if sweep_id is provided.')

    parser.add_argument('--project', type=str, default='AblateIt-Sweeps',
                        help='Wandb project name. Do not change.')

    parser.add_argument('--default_training_args', type=str,
                        default='configs/default_training_configs/default_qlora.yaml',
                        help='Path to default training args yaml file. Ignored if sweep_id is provided.')

    parser.add_argument('--entity', type=str, default='ablateit',
                        help='Wandb entity name. Do not change unless testing.')

    parser.add_argument('--push_to_hub', type=bool, default=True,
                        help='Whether to push the models to the hub during training.')

    # parser.add_argument('--dataset', type=str, default='LDJnr/Puffin',
    #                     help='Dataset to use for training. Currently only supports Puffin.')

    return parser.parse_args()


DATASET_SIZES = {"Puffin" : 3000}


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


def sweep():
    args = get_args()

    temp_config_path = args.default_training_args.replace('.yaml', '_temp.yaml')
    shutil.copyfile(args.default_training_args, temp_config_path)

    sweep_id = args.sweep_id

    if not sweep_id:
        sweep_config = yaml.safe_load(open(args.sweep_config))["wandb_args"]
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(sweep_id)
        with open("sweep_id.txt", "w") as file:
            file.write(sweep_id)

    def run_sweep():
        wandb.init(entity=args.entity)
        config = dict(wandb.config)

        warmup_factor = config.pop(
            "warmpup_steps_factor_of_epoch") if "warmpup_steps_factor_of_epoch" in config else None
        finetune_type = config.pop("ft_type")
        sweep_name = config.pop("sweep_name")

        run_name = sweep_name + "-" + finetune_type + "-" + create_name(config)

        wandb.run.name = run_name
        with open(temp_config_path, 'r') as file:
            temp_config = yaml.safe_load(file)

        for hyperparameter, value in config.items():
            temp_config[hyperparameter] = value

        if warmup_factor:
            temp_config["warmup_steps"] = int((DATASET_SIZES["Puffin"] * (1 - temp_config["val_set_size"] ) )\
                        / (temp_config["gradient_accumulation_steps"] * temp_config["micro_batch_size"]) * warmup_factor)

        if args.push_to_hub:
            temp_config["hub_model_id"] = "AblateIt/" + run_name
            temp_config["push_to_hub"] = True
            temp_config["hub_strategy"] = "all_checkpoints"
            print(temp_config["hub_model_id"])

        temp_config["wandb_project"] = "AblateIt-Sweeps"
        temp_config["wandb_entity"] = args.entity
        temp_config["wandb_run_name"] = run_name
        temp_config["output_dir"] = temp_config["output_dir"] + '/' + run_name + '/'

        run_config_path = temp_config["output_dir"] + '/config.yaml'

        if not os.path.exists(temp_config["output_dir"]):
                    os.makedirs(temp_config["output_dir"])

        with open(run_config_path, 'w') as file:
            yaml.dump(temp_config, file)

        # Run the training command with the temporary config file
        cuda_device_declaration = "CUDA_VISIBLE_DEVICES=" + ",".join(
            [str(x) for x in args.CUDA_device_ids]) + " " if args.CUDA_device_ids else ""
        cmd = cuda_device_declaration + f"accelerate launch axolotl/scripts/finetune.py {run_config_path}"
        call(cmd, shell=True)

    # Run the sweep
    wandb.agent(sweep_id, run_sweep, project=args.project, entity=args.entity)

    os.remove(temp_config_path)



if __name__ == '__main__':
    sweep()
