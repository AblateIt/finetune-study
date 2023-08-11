## Structure
### sweep_configs
Contains configuration files for each sweep:
- "full_ft_sweep.yaml" - Full Fine-Tuning Sweep
- "lora_sweep.yaml" - LoRA Sweep
- "qlora_sweep.yaml" - QLoRA Sweep

### default_training_args
Contains the default training arguments for each fine-tuning method taken directly from axolotl:
- "default_lora.yaml" - LoRA Default Training Arguments
- "default_qlora.yaml" - QLoRA Default Training Arguments

### test
Contains the Puffin Llama 2 7b configuration mainly used for testing
- "qlora_experiment.yaml" 

##  Notes
- Currently, only lora_sweep.yaml has the most context for making sweep configurations. It also needs to be updated, TODO's are in the file.
- The default training arguments for each fine-tuning method are taken directly from axolotl, so they need to be updated to fit our needs and project structure.
- There is no default training arguments file for full fine-tuning, so we need to create one.
