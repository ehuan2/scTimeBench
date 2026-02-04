"""
Note: for this file only, this will be used by other models as a base class
And so its context is outside the src/ folder, so we need to use crispy_fishstick.*
imports instead of relative imports.
"""

import argparse
import os
import pickle
import yaml
import scanpy as sc
from scipy.sparse import issparse
import numpy as np

from crispy_fishstick.shared.constants import RequiredOutputColumns
from crispy_fishstick.shared.constants import ObservationColumns


def get_parser():
    # parser that will read the input data path and the model output path
    parser = argparse.ArgumentParser(description="Train ExampleRandomSampler model.")
    parser.add_argument(
        "--yaml_config", type=str, help="Path to YAML configuration file"
    )
    return parser


def process_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    paths = ["dataset_pkl_path", "output_path", "output_file_name"]
    for path in paths:
        if path not in yaml_config:
            raise ValueError(f"YAML configuration missing required path: {path}")

        # verify the data paths exist:
        if path in [
            "output_path",
            "dataset_pkl_path",
        ] and not os.path.exists(yaml_config[path]):
            raise FileNotFoundError(f"Data file not found: {yaml_config[path]}")

    # now let's load the dataset from the pickled path
    with open(yaml_config["dataset_pkl_path"], "rb") as f:
        yaml_config["dataset"] = pickle.load(f)

    return yaml_config


# Class to be inherited by all models
class BaseModel:
    def __init__(self, yaml_config):
        self.config = yaml_config

        # normalize required outputs: list or list of lists
        raw_required_outputs = self.config["required_outputs"]
        if not raw_required_outputs:
            raise ValueError("required_outputs must not be empty.")

        if all(isinstance(item, list) for item in raw_required_outputs):
            required_output_options = [
                [RequiredOutputColumns(output) for output in option]
                for option in raw_required_outputs
            ]
        else:
            required_output_options = [
                [RequiredOutputColumns(output) for output in raw_required_outputs]
            ]

        self.required_outputs_options = required_output_options

        # by default since we are not an OT method, we just select the option without NEXT_CELLTYPE
        for option in required_output_options:
            if RequiredOutputColumns.NEXT_CELLTYPE not in option:
                self.required_outputs = option
                break

        print(f"Required outputs: {self.required_outputs}")

    def train(self, ann_data, all_tps=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def generate(self, test_ann_data, expected_output_path):
        raise NotImplementedError("Subclasses should implement this method.")


class BaseOTModel(BaseModel):
    """
    Base class for OT-based models.
    """

    def __init__(self, yaml_config):
        super().__init__(yaml_config)

        # select the option that has NEXT_CELLTYPE if it's an OT method
        for option in self.required_outputs_options:
            if RequiredOutputColumns.NEXT_CELLTYPE in option:
                self.required_outputs = option
                break

        if not hasattr(self, "required_outputs"):
            print(
                f"Warning: OT method but no NEXT_CELLTYPE in required outputs. Using first option."
            )
            self.required_outputs = self.required_outputs_options[0]

        print(f"Required outputs for OT method: {self.required_outputs}")

    def train(self, ann_data, all_tps=None):
        print(
            f"OT-based model training not part of OT paradigm. Skipping training step."
        )

    def _prepare_generate(self, test_ann_data):
        """
        By default we don't need to do anything here. Subclasses representing OT methods can override this method if needed.
        """

    def generate(self, test_ann_data, expected_output_path):
        self._prepare_generate(test_ann_data)
        self.ot_generate(test_ann_data, expected_output_path)

    def get_transport_plan(self, source_data, target_data):
        """
        Given source and target data, compute the transport plan.
        Subclasses representing OT methods should implement this method.

        Parameters:
        -----------
        source_data : np.ndarray
            Source data matrix (cells x features)
        target_data : np.ndarray
            Target data matrix (cells x features)

        Returns:
        --------
        np.ndarray
            Transport plan matrix (source cells x target cells)
        """
        raise NotImplementedError(
            "Subclasses representing OT methods should implement this method."
        )

    def ot_generate(self, test_ann_data, expected_output_path):
        """
        Generate method for OT-based methods. Subclasses representing OT methods should provide a get_transport_plan method.
        """
        print(f"Generating using OT-based method...")
        final_ann_data = test_ann_data.copy()

        # Get timepoint information
        test_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
        unique_tps = sorted(np.unique(test_tps))

        require_next_tp = (
            RequiredOutputColumns.NEXT_CELLTYPE in self.required_outputs
            or RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION
            in self.required_outputs
            or RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING in self.required_outputs
        )

        final_ann_data.obsm[RequiredOutputColumns.NEXT_CELLTYPE.value] = np.full(
            (test_ann_data.n_obs,), "unknown", dtype=object
        )
        final_ann_data.obsm[
            RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value
        ] = np.full((test_ann_data.n_obs, test_ann_data.n_vars), np.nan, dtype=float)

        if require_next_tp:
            # Process each timepoint transition
            for i, tp in enumerate(unique_tps[:-1]):
                print(f"Processing timepoint transition: {tp} -> {unique_tps[i + 1]}")
                next_tp = unique_tps[i + 1]

                # Solve temporal problem for this transition
                transport_plan = self.get_transport_plan(test_ann_data, tp, next_tp)

                source_mask = (
                    test_ann_data.obs[ObservationColumns.TIMEPOINT.value] == tp
                )
                dest_mask = (
                    test_ann_data.obs[ObservationColumns.TIMEPOINT.value] == next_tp
                )
                assert transport_plan.shape == (
                    source_mask.sum(),
                    dest_mask.sum(),
                ), f"Transport plan shape {transport_plan.shape} does not match the number of source and destination cells ({source_mask.sum()}, {dest_mask.sum()})"

                if RequiredOutputColumns.NEXT_CELLTYPE in self.required_outputs:
                    inferred_cell_types = self.get_next_cell_type_from_transport_plan(
                        transport_plan, final_ann_data, dest_mask
                    )

                    # Assign inferred cell types to the appropriate cells
                    # i.e. we want to assign to cells at timepoint tp
                    for j, cell_idx in enumerate(np.where(source_mask)[0]):
                        # overwrite the obsm with the inferred cell type
                        final_ann_data.obsm[RequiredOutputColumns.NEXT_CELLTYPE.value][
                            cell_idx
                        ] = inferred_cell_types[j]

                if (
                    RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION
                    in self.required_outputs
                ):
                    # Assign inferred gene expression to the appropriate cells
                    inferred_gex = self.get_next_cell_gex_from_transport_plan(
                        transport_plan, final_ann_data, dest_mask
                    )
                    for j, cell_idx in enumerate(np.where(source_mask)[0]):
                        final_ann_data.obsm[
                            RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value
                        ][cell_idx, :] = inferred_gex[j]

            if RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING in self.required_outputs:
                # Compute embeddings for the next timepoint gene expression
                # this automatically handles the embedding computation as well
                # ** Note: need to plug in final_ann_data since it needs to be updated **
                pca_model = self.get_embedding_from_gex(final_ann_data)
                self.get_next_embedding_from_gex(final_ann_data, pca_model)

        elif RequiredOutputColumns.EMBEDDING in self.required_outputs:
            # ** Note: need to plug in final_ann_data since it needs to be updated **
            self.get_embedding_from_gex(final_ann_data)

        # Write output
        print(f"Writing output to {expected_output_path}")
        final_ann_data.write_h5ad(expected_output_path)
        print("Generation complete.")

    def get_next_cell_type_from_transport_plan(
        self, transport_plan, ann_data, dest_mask
    ) -> list:
        """
        Given a transport plan, the anndata, the source and destination time points, infer the next cell types.

        Parameters:
        -----------
        transport plan: a matrix that is source by destination cells
        ann_data: anndata object containing the cell data
        source_tp: source time point
        dest_tp: destination time point

        Returns:
        --------
        1. list of inferred next cell types for each source cell
        2. list of inferred next cell expression for each source cell
        """
        dest_cell_types = ann_data.obs[ObservationColumns.CELL_TYPE.value].to_numpy()[
            dest_mask
        ]

        # now let's iterate through each source cell and infer its next cell type
        target_cell_types = []
        total_no_weight = 0

        # let's modify this to be matrix operations
        # 1. we need to get the one hot encoding of the dest cell types
        # 2. then we can do a matrix multiplication to get the distribution for each source to dest cell type
        # using source x dest multiplied by dest x cell type one hot encoding
        # 3. then we can get the argmax for each source cell to get the target cell type
        # 4. for expected expression, we can do source x dest multiplied by dest x gene expression
        # which gives us source x gene expression directly

        # 1. one hot encoding of dest cell types
        unique_cell_types = sorted(np.unique(dest_cell_types))
        cell_type_to_index = {ct: i for i, ct in enumerate(unique_cell_types)}
        one_hot_dest = np.zeros((len(dest_cell_types), len(unique_cell_types)))
        for i, ct in enumerate(dest_cell_types):
            one_hot_dest[i, cell_type_to_index[ct]] = 1.0
        print(
            f"Cell type to index: {cell_type_to_index}, one hot shape: {one_hot_dest.shape}"
        )

        # 2. matrix multiplication to get distribution
        cell_type_distribution = (
            transport_plan @ one_hot_dest
        )  # shape: source cells x cell
        # 3. argmax to get target cell type
        # ** Note: we calculate the next cell type distribution based on weights **
        # ** Where we aggregate the probabilities for each cell type based on the transport weights **
        for i in range(cell_type_distribution.shape[0]):
            distribution = cell_type_distribution[i, :]
            if np.sum(distribution) == 0:
                total_no_weight += 1
            target_cell_types.append(unique_cell_types[np.argmax(distribution)])

        if total_no_weight > 0:
            print(
                f"Warning: {total_no_weight} source cells had zero transport weights to destination cells. This may become a source of bias."
            )

        return target_cell_types

    def get_next_cell_gex_from_transport_plan(
        self, transport_plan, ann_data, dest_mask
    ):
        data = ann_data.X.toarray() if issparse(ann_data.X) else ann_data.X
        dest_data = data[dest_mask, :]

        # TODO: this does not work the suo dataset because it's too large for memory
        # TODO: need to implement a batching mechanism here to handle large datasets
        target_cell_expression = (
            transport_plan @ dest_data
        )  # shape: source cells x genes

        return target_cell_expression

    def get_embedding_from_gex(self, ann_data, n_comps=50):
        """
        Given an anndata object, compute the PCA embedding from the gene expression data.

        Parameters:
        -----------
        ann_data: anndata object containing the cell data

        Returns:
        --------
        PCA model.

        Note: the embedding is stored in ann_data.obsm[RequiredOutputColumns.EMBEDDING.value]
        """
        print(f"Computing PCA embedding from gene expression data...")
        pca_model = sc.pp.pca(ann_data, n_comps=n_comps)
        # rename the X_pca to embedding for consistency
        ann_data.obsm[RequiredOutputColumns.EMBEDDING.value] = ann_data.obsm.pop(
            "X_pca"
        )
        return pca_model

    def get_next_embedding_from_gex(self, ann_data, pca_model):
        """
        Given an anndata object and a PCA model, compute the PCA embedding from the gene expression data.

        Parameters:
        -----------
        ann_data: anndata object containing the cell data
        pca_model: PCA model to use for transformation

        Returns:
        --------
        None. The embedding is stored in ann_data.obsm[RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value]
        """
        gex_data = ann_data.obsm[
            RequiredOutputColumns.NEXT_TIMEPOINT_GENE_EXPRESSION.value
        ]
        embedding = pca_model.transform(gex_data)
        ann_data.obsm[RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value] = embedding


def main(model_class: BaseModel):
    print(f"Starting train and testing for model...")
    parser = get_parser()
    args = parser.parse_args()
    yaml_config = process_yaml(args.yaml_config)

    # if the model outputs already exist, then we skip generation
    model_output_path = os.path.join(
        yaml_config["output_path"], yaml_config["output_file_name"]
    )
    if os.path.exists(model_output_path):
        print(
            f'Generated samples found at {os.path.join(yaml_config["output_path"], yaml_config["output_file_name"])}. Skipping generation.'
        )
        return

    # Otherwise we have to load the data and train/test the model
    print("Loading dataset...")
    train_ann_data, test_ann_data = yaml_config["dataset"].load_data()

    # Some methods map the tps to indices, argument all used for pertinent methods.
    # Providing it to train argument for processing to be handled within the subclasses.
    time_col = ObservationColumns.TIMEPOINT.value
    all_tps = (
        train_ann_data.obs[time_col].unique().tolist()
        + test_ann_data.obs[time_col].unique().tolist()
    )
    all_tps = list(set(all_tps))

    # Initialize the model
    model: BaseModel = model_class(yaml_config)

    print(f"Training and/or loading the model: {model_class.__name__}")
    # let's let the train() function handle the caching as well
    model.train(train_ann_data, all_tps=all_tps)
    print("Training/loading complete.")

    # Generate samples -- we'll move the saving of generated samples outside of this script
    print(f"Starting generation to {model_output_path}")
    model.generate(test_ann_data, expected_output_path=model_output_path)
    print("Generation complete.")

    # verify that the output file was created, and that it contains the required columns
    if not os.path.exists(model_output_path):
        raise RuntimeError(f"Model output file was not created at {model_output_path}")
    print(f"Verifying generated output at {model_output_path}")

    # TODO: add generate and train under a try catch which will clean the model output path if anything fails
    # load the ann data and check for required columns
    generated_ann_data = sc.read_h5ad(model_output_path)
    for required_output in model.required_outputs:
        if required_output.value not in generated_ann_data.obsm.keys():
            # delete the generated file to avoid confusion
            os.remove(model_output_path)
            raise RuntimeError(
                f"Generated output missing required column: {required_output.value}"
            )
