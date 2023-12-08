import argparse
import pathlib
from sim_clr.sim_clr_test import run_self_supervised_testing
from sim_clr.sim_clr_train import train_sim_clr

path_to_data = "../fruit/Dataset/Grading_dataset"
path_to_weights = '../models/sim_clr_res18_e(50)_256x256_mango.pth'

def parse_args():
    parser = argparse.ArgumentParser(description='Self-supervised learning model training and testing')

    # Training parameters
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--input_size', type=int, default=256, help='Input size for the model')

    # Training output
    parser.add_argument('--weights_file', type=str, default='../models/', help='Output file path for training weights')


    # Model and dataset parameters
    parser.add_argument('--dataset_train', type=str, default=path_to_data, help='Path to training dataset to use')
    parser.add_argument('--dataset_test', type=str, default=path_to_data, help='Path to testing dataset to use')
    parser.add_argument('--model', type=str, default="SimCLR", help='Model to use')

    # Testing mode
    parser.add_argument('--testing', action='store_true', help='Enable testing mode (ignore other training args)')
    parser.add_argument('--weights',default=path_to_weights,help='Path to pretrained weights')

    # Output directory for testing mode
    parser.add_argument('--output', type=str, default='../results/', help='Output directory for testing mode')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.testing:
        print(f'Testing mode enabled. Using dataset: {args.dataset_test}, model: {args.model} weights: {args.weights}, output directory: {args.output}')
        # Add testing logic here
        args.output = pathlib.Path(args.output)
        if args.model == "SimCLR":
            run_self_supervised_testing(args)
        else:
            raise NotImplementedError("Have not implemented testing for DINO yet")

    else:
        print(f'Training mode enabled. Using dataset: {args.dataset_test}, model: {args.model}')
        print(f'Training parameters - num_workers: {args.num_workers}, batch_size: {args.batch_size}, seed: {args.seed}, max_epochs: {args.max_epochs}, input_size: {args.input_size}')
        # Add training logic here
        if args.model == "SimCLR":
            train_sim_clr(args.num_workers,args.batch_size,args.weights_file,args.dataset_train,args.seed,args.max_epochs,args.input_size,)
        else:
            raise NotImplementedError("Have not implemented training for DINO yet")




if __name__ == "__main__":
    main()
