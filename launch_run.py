from sweep import get_args, create_name, train, DATASET_SIZES

if __name__ == "__main__":
    args = get_args()
    print(vars(args))
    train(args)