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
