import argparse
import scanpy as sc
import pickle


def get_parser():
    # parser that will read the input data path and the model output path
    parser = argparse.ArgumentParser(description="Train ExampleRandomSampler model.")
    parser.add_argument("--train_data_path", type=str, help="Path to input data file")
    parser.add_argument("--test_data_path", type=str, help="Path to input data file")
    parser.add_argument(
        "--model_output_path", type=str, help="Path to save the trained model"
    )
    parser.add_argument(
        "--train", action="store_true", help="Flag to indicate training mode"
    )
    parser.add_argument(
        "--test", action="store_true", help="Flag to indicate training mode"
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save generated samples"
    )
    return parser


def main(model_class):
    parser = get_parser()
    args = parser.parse_args()

    if args.train:
        # Load data
        ann_data = sc.read_h5ad(args.train_data_path)

        # Initialize and train model
        model = model_class()
        model.train(ann_data)

        # Save the trained model
        with open(args.model_output_path, "wb") as f:
            pickle.dump(model, f)

    if args.test:
        model = model_class()
        # Load the trained model
        with open(args.model_output_path, "rb") as f:
            model = pickle.load(f)

        # Load test data
        test_ann_data = sc.read_h5ad(args.test_data_path)

        # Generate samples
        generated_samples = model.generate(test_ann_data)

        # Save generated samples
        with open(args.output_path, "wb") as f:
            pickle.dump(generated_samples, f)
