import wandb
import argparse
import yaml
import shutil
from subprocess import call
import os

wandb.login()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Wandb sweep id for decentralized sweeping. If not provided, a new sweep will be created.",
    )

    parser.add_argument(
        "--gpu",
        type=list,
        default=None,
        help="List of CUDA device ids to use for training. If not provided, all available GPUs will be used.",
    )

    parser.add_argument(
        "--sweep_config",
        type=str,
        default="configs/sweep_configs/qlora_sweep.yaml",
        help="Path to sweep config yaml file. Ignored if sweep_id is provided.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="AblateIt-Sweeps",
        help="Wandb project name. Do not change.",
    )

    parser.add_argument(
        "--default_training_args",
        type=str,
        default="configs/default_training_configs/default_qlora.yaml",
        help="Path to default training args yaml file. Ignored if sweep_id is provided.",
    )

    parser.add_argument(
        "--entity",
        type=str,
        default="ablateit",
        help="Wandb entity name. Do not change unless testing.",
    )

    parser.add_argument(
        "--push_to_hub",
        type=bool,
        default=True,
        help="Whether to push the models to the hub during training.",
    )

    # parser.add_argument('--dataset', type=str, default='LDJnr/Puffin',
    #                     help='Dataset to use for training. Currently only supports Puffin.')

    return parser.parse_args()


DATASET_SIZES = {"Puffin": 3000}


def create_name(config_dict):
    short = {
        "gradient_accumulation_steps": "graccsteps",
        "learning_rate": "lr",
        "lora_r": "lora_r",
        "lora_dropout": "drop",
    }
    name = ""
    for hyperparam, value in config_dict.items():
        name += short.get(hyperparam, hyperparam) + str(value).replace(".", "_") + "-"
    return name[:-1]


def sweep():
    args = get_args()

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

        warmup_factor = (
            config.pop("warmpup_steps_factor_of_epoch")
            if "warmpup_steps_factor_of_epoch" in config
            else None
        )
        finetune_type = config.pop("ft_type")
        sweep_name = config.pop("sweep_name")

        run_name = args.project + "-" + sweep_name + "-" + finetune_type + "-" + create_name(config)

        wandb.run.name = run_name
        with open(args.default_training_args, "r") as file:
            run_config = yaml.safe_load(file)

        for hyperparameter, value in config.items():
            run_config[hyperparameter] = value

        epoch_train_steps = int((DATASET_SIZES["Puffin"] *
                               (1 - run_config["val_set_size"])) / (run_config["gradient_accumulation_steps"] * run_config["micro_batch_size"]))

        if warmup_factor:
            run_config["warmup_steps"] = int(epoch_train_steps * warmup_factor)

        if run_config["eval_strategy"] == "epoch" and type(run_config["eval_steps"]) == float:
            run_config["eval_steps"] = int(epoch_train_steps * run_config["eval_steps"])
            run_config["eval_strategy"] = "steps"

        if run_config["save_strategy"] == "epoch" and type(run_config["save_steps"]) == float:
            run_config["save_steps"] = int(epoch_train_steps * run_config["save_steps"])
            run_config["save_strategy"] = "steps"

        if args.push_to_hub:
            run_config["hub_model_id"] = "AblateIt/" + run_name
            run_config["push_to_hub"] = True
            run_config["hub_strategy"] = "all_checkpoints"
            print(run_config["hub_model_id"])

        run_config["wandb_project"] = args.project
        run_config["wandb_entity"] = args.entity
        run_config["wandb_run_name"] = run_name
        run_config["output_dir"] = run_config["output_dir"] + "/" + run_name + "/"

        run_config_path = run_config["output_dir"] + "config.yaml"

        if not os.path.exists(run_config["output_dir"]):
            os.makedirs(run_config["output_dir"])

        with open(run_config_path, "w") as file:
            yaml.dump(run_config, file)
        print(run_config)

        # Run the training command with the temporary config file
        cuda_device_declaration = (
            "export CUDA_VISIBLE_DEVICES=" + ",".join([str(x) for x in args.gpu]) + "; "
            if args.gpu
            else ""
        )
        cmd = (
            cuda_device_declaration
            + f"accelerate launch axolotl/scripts/finetune.py {run_config_path} --main_process_port 0"
        )
        print(cmd)
        call(cmd, shell=True)

    if args.sweep_id is not None:
        # Run the sweep
        wandb.agent(sweep_id, run_sweep, project=args.project, entity=args.entity)


if __name__ == "__main__":
    sweep()
