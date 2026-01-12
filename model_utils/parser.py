import argparse
import scanpy as sc
import pickle
import os
import yaml


def get_parser():
    # parser that will read the input data path and the model output path
    parser = argparse.ArgumentParser(description="Train ExampleRandomSampler model.")
    parser.add_argument(
        "--yaml_config", type=str, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--train", action="store_true", help="Flag to indicate training mode"
    )
    parser.add_argument(
        "--test", action="store_true", help="Flag to indicate training mode"
    )
    return parser


def process_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    paths = ["train_data_path", "test_data_path", "output_path", "output_file_name"]
    for path in paths:
        if path not in yaml_config:
            raise ValueError(f"YAML configuration missing required path: {path}")

        # verify the data paths exist:
        if path in [
            "train_data_path",
            "test_data_path",
            "output_path",
        ] and not os.path.exists(yaml_config[path]):
            raise FileNotFoundError(f"Data file not found: {yaml_config[path]}")

    return yaml_config


# Class to be inherited by all models
class BaseModel:
    def __init__(self):
        pass

    def train(self, ann_data):
        raise NotImplementedError("Subclasses should implement this method.")

    def generate(self, ann_data):
        raise NotImplementedError("Subclasses should implement this method.")


def main(model_class: BaseModel):
    parser = get_parser()
    args = parser.parse_args()
    yaml_config = process_yaml(args.yaml_config)

    if args.train:
        # first we check to see if the checkpointed model exists
        if os.path.exists(os.path.join(yaml_config["output_path"], "model.pkl")):
            print(
                f'Checkpointed model found at {os.path.join(yaml_config["output_path"], "model.pkl")}. Skipping training.'
            )
            # we don't exit here because the user may want to run testing right after
        else:
            # Load data
            ann_data = sc.read_h5ad(yaml_config["train_data_path"])

            # Initialize and train model
            model = model_class()
            print(f"Training model: {model_class.__name__}")
            model.train(ann_data)
            print("Training complete.")

            # Save the trained model
            print(f'Saving trained model to {yaml_config["output_path"]}/model.pkl')
            with open(os.path.join(yaml_config["output_path"], "model.pkl"), "wb") as f:
                pickle.dump(model, f)

    if args.test:
        model = model_class()
        # Load the trained model
        print(f'Loading trained model from {yaml_config["output_path"]}/model.pkl')
        with open(os.path.join(yaml_config["output_path"], "model.pkl"), "rb") as f:
            model = pickle.load(f)

        # Load test data
        test_ann_data = sc.read_h5ad(yaml_config["test_data_path"])
        # Generate samples -- we'll move the saving of generated samples outside of this script
        model.generate(
            test_ann_data,
            os.path.join(yaml_config["output_path"], yaml_config["output_file_name"]),
        )
