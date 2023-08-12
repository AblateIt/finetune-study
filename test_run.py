import argparse
import yaml
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_args_path', type=str, help='Path to default training args yaml file',
                        default='configs/default_training_configs/default_qlora.yaml')

    return parser.parse_args()

def main():
    args = get_args()

    with open(args.training_args_path, 'r') as file:
        temp_config = yaml.safe_load(file)

    print(temp_config)

if __name__ == '__main__':
    main()