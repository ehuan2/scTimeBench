from scTimeBench.metrics.meta.base import MetaMetric
from scTimeBench.shared.utils import load_train_dataset
from scTimeBench.shared.dataset.base import BaseDataset

import scanpy as sc
import pandas as pd
import numpy as np

import logging
import os


class GRN:
    def __init__(self, grn_path):
        if not os.path.exists(grn_path):
            raise FileNotFoundError(f"GRN file not found at {grn_path}")

        # check the format, if it's tsv, read it as a dataframe
        if grn_path.endswith(".tsv"):
            self.grn_df = pd.read_csv(grn_path, sep="\t")
            self.tf_col = "TF"
            self.gene_col = "Target"
            self.regulation = "Regulation"

    def get_genes_from_tf(self, tf):
        """
        Get the transcription factor (TF) and gene pairs from the GRN dataframe,
        given a specific TF.

        Returns:
            List of genes regulated by the given TF, and direction if available.
        """
        if tf not in self.grn_df[self.tf_col].values:
            return []

        tf_genes = self.grn_df[self.grn_df[self.tf_col] == tf][
            [self.gene_col, self.regulation]
        ]
        return tf_genes.to_dict("records")

    def get_tfs_from_gene(self, gene):
        """
        Get the transcription factor (TF) and gene pairs from the GRN dataframe,
        given a specific gene.

        Returns:
            List of TFs that regulate the given gene, and direction if available.
        """
        if gene not in self.grn_df[self.gene_col].values:
            return []

        gene_tfs = self.grn_df[self.grn_df[self.gene_col] == gene][
            [self.tf_col, self.regulation]
        ]
        return gene_tfs.to_dict("records")


class MetaGRN(MetaMetric):
    """
    Meta submetric for GRN analyses.
    """

    def _defaults(self):
        """The default parameters for grn-meta-based metrics."""
        return {
            "grn_path": "grn_data/resources/trrust_rawdata.human.tsv",
            "genes": None,
            "num_genes": 5,
            "gene_col_name": None,
        }

    def _get_highly_variable_genes(self, output_path, n_top_genes):
        """Get the top n highly variable genes from the dataset."""
        train_dataset = load_train_dataset(output_path)
        # get the highly variable genes
        sc.pp.highly_variable_genes(
            train_dataset, n_top_genes=n_top_genes, flavor="seurat"
        )
        hvg_df = train_dataset.var[train_dataset.var["highly_variable"] == True]
        top_n_genes = hvg_df.index.tolist()

        print(f"Top {n_top_genes} highly variable genes: {top_n_genes}")

        top_n_genes_names = (
            hvg_df[self.params["gene_col_name"]].tolist()
            if self.params["gene_col_name"]
            else top_n_genes
        )
        logging.debug(
            f"Top {n_top_genes} highly variable genes with names: {top_n_genes_names}"
        )

        def debug_hvgs():
            # let's print the std to show that these are indeed highly variable genes
            hvg_mask = train_dataset.var["highly_variable"] == True

            # --- Helper function to extract matrix math safely ---
            def get_matrix_metrics(adata_subset):
                if hasattr(adata_subset.X, "toarray"):
                    matrix = adata_subset.X.toarray()
                else:
                    matrix = adata_subset.X
                return matrix.mean(axis=0), matrix.std(axis=0)

            # --- 1. Calculate metrics for Highly Variable Genes ---
            hvg_adata = train_dataset[:, hvg_mask]
            hvg_means, hvg_stds = get_matrix_metrics(hvg_adata)

            logging.debug("\n--- Highly Variable Genes Expression Metrics (Top 10) ---")
            for i, gene in enumerate(top_n_genes_names[:10]):
                logging.debug(
                    f"Gene: {gene:12} | Mean Expression: {hvg_means[i]:.4f} | Std Dev: {hvg_stds[i]:.4f}"
                )
            if len(top_n_genes_names) > 10:
                logging.debug(f"... and {len(top_n_genes_names) - 10} more HVGs.")

            # --- 2. EXTENSION: Calculate metrics for 10 Random Other Genes ---
            # Get dataframe of genes that are NOT marked as highly variable
            non_hvg_df = train_dataset.var[~hvg_mask]

            if len(non_hvg_df) >= 10:
                # Randomly sample 10 indices from the non-HVG pool
                random_indices = np.random.choice(
                    non_hvg_df.index, size=10, replace=False
                )
                random_subset_df = non_hvg_df.loc[random_indices]

                # Map to common names if the parameter is provided
                random_genes_names = (
                    random_subset_df[self.params["gene_col_name"]].tolist()
                    if self.params["gene_col_name"]
                    else random_indices.tolist()
                )

                # Slice dataset and compute stats for the random genes
                random_adata = train_dataset[:, random_indices]
                random_means, random_stds = get_matrix_metrics(random_adata)

                logging.debug(
                    "\n--- Random Baseline Genes Expression Metrics (10 Random) ---"
                )
                for i, gene in enumerate(random_genes_names):
                    logging.debug(
                        f"Gene: {gene:12} | Mean Expression: {random_means[i]:.4f} | Std Dev: {random_stds[i]:.4f}"
                    )
            else:
                logging.debug(
                    "\n[Warning] Not enough non-highly variable genes available to sample baseline."
                )

            logging.debug("------------------------------------------------\n")

        if self.config.log_level == "DEBUG":
            debug_hvgs()

        return top_n_genes_names

    def _submetric_eval(self, output_path, dataset: BaseDataset, method):
        """
        Run the submetric evaluation for GRN analyses.

        Args:
            output_path (str): The path to the output directory.
            dataset (Dataset): The dataset object.
            method (Method): The method object.
        """
        self.grn = GRN(grn_path=self.params["grn_path"])

        # Then we need to select the genes to perturb.
        # We either select the genes chosen in the parameters,
        # or select the top n highly variable genes in the dataset.
        genes = self.params["genes"]
        if genes is None or len(genes) == 0:
            genes = self._get_highly_variable_genes(
                output_path, self.params["num_genes"]
            )

        exit()
        # Now we need to first choose one of the more important genes
        # i.e. a highly variable gene, and then figure out either:
        # 1) the genes that regulate it
        # 2) the genes that it regulates
        # 3) a random gene that isn't related to the gene
        # And we choose 5 genes from either category
        # to up and down regulate and calculate the perturbation

        # To do this, we will do t to t + 1 for all the cells
        # and perturb it in this way instead
