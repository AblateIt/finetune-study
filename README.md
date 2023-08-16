# Comparing QLora, Lora, and Full Fine-tuning
Comprehensive analysis of difference in performance of QLora, Lora, and Full Fine-tuning.


## Installation
### 1. Install python 3.9.x or 3.10.x
### 2. Install pytorch stable:
>https://pytorch.org/get-started/locally/
>
>If this step fails, either at install or gives an error when training, do `pip uninstall torch` and try simply `pip install torch`
>
### 3. Install axolotl and dependencies
```
git clone https://github.com/AblateIt/axolotl.git
pip3 install -e axolotl/.
pip3 install -U git+https://github.com/huggingface/peft.git
```
There is a `requirements.txt` file in this repo, you might need to install some packages from this depending on what you are missing.

## For contributors running sweeps and training
### 1. Request access to the AblateIt WandB and HuggingFace teams
### 2. Log into wandb and HuggingFace through the CLI
    wandb login (login with the account added to the wandb org)
    huggingface-cli login (login with the account added to the HF org)

### How to start a sweep (you most likely will never do it)
1. Activate the correct environment
2. Set the default location to create new projects to `ablateit`. This is required to create the sweep but not to run finetuning.
3. `python sweep.py --sweep_config <path_to_sweep_config> --project <wandb_project_name> --default_training_args <default_config_file_for_experiment>`

For example to run QLora sweep, this command can be run
`python sweep.py --sweep_config configs/sweep_configs/qlora_sweep.yaml --project test-qlora_sweep --default_training_args configs/default_training_configs/default_qlora.yaml`

### How to Finetune configurations from a sweep.
1. Check if you have a default acclerate config and if you have it then delete it. You can check your huggingface cache folder, by default it points to this `~/.cache/huggingface/accelerate/default_config.yaml`, if the `default_config.yaml` file exists then delete it.
2. Test your code by running the command `CUDA_VISIBLE_DEVICES=0 accelerate launch axolotl/scripts/finetune.py configs/test/qlora_experiment.yaml --main_process_port 0`, this should run a qlora run on your GPU0. If not then please fix the error before running a sweep or else you will pull configurations from the sweep which will crash and no one else will be able to run them as well.

3. You would need a `sweep_id` and a `project_id` from one of the contributor who has started a sweep in order to run finetune experiments.

`python sweep.py --sweep_id <sweep_id> --project <project_id> --gpu <gpu_id>`

For example, this sample command will run finetuning on GPU 0.
`python sweep.py --sweep_id usevjjyj  --CUDA_device_ids 0`


## FAQs
#### 1. Accelerate running experiments on multiple GPUs or other accelerate issues.
Go to your huggingface cache folder and delete the `default_config.yaml` file. For examples the default location of this file would be would be at `~/.cache/huggingface/accelerate/default_config.yaml`.

When running finetuning, if you are **NOT** seeing a messge like this, then you have a default accelerate config that is saved in your cache that needs to be **DELETED**.
```python
The following values were not passed to `accelerate launch` and had defaults used instead:
        `--num_processes` was set to a value of `1`
        `--num_machines` was set to a value of `1`
        `--mixed_precision` was set to a value of `'no'`
        `--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
```

## Links
- [Discord](https://discord.gg/HfNctSTJ)
- [HuggingFace](https://huggingface.co/AblateIt)
- [WandB](https://wandb.ai/ablateit)
