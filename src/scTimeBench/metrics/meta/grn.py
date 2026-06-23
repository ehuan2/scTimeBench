from scTimeBench.metrics.meta.base import MetaMetric
from scTimeBench.shared.utils import load_train_dataset
from scTimeBench.shared.constants import ObservationColumns
from scTimeBench.shared.dataset.base import BaseDataset
from collections import deque
from scipy.sparse import issparse

import scanpy as sc
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

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

    def get_all_genes(self):
        """
        Get all unique genes in the GRN, including both TFs and target genes.

        Returns:
            Set of all unique genes in the GRN.
        """
        tf_genes = set(self.grn_df[self.tf_col].unique())
        target_genes = set(self.grn_df[self.gene_col].unique())
        all_genes = tf_genes.union(target_genes)
        return all_genes

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

    def get_full_graph(self, start_tf, choose_target):
        """
        Traverses the Gene Regulatory Network downstream from a starting TF
        using Breadth-First Search (BFS).

        Args:
            start_tf (str): The initial transcription factor to start the
            cascade from.
            choose_target (bool): If True, traverse downstream to find target
                                    genes. If False, traverse upstream to find
                                    regulating TFs.

        Returns:
            List[dict]: A list of directed edges representing the full cascade
                        network. Each item format: {'TF': ..., 'Target': ...,
                        'Regulation': ...}
        """
        start_col = self.tf_col if choose_target else self.gene_col
        target_col = self.gene_col if choose_target else self.tf_col

        # If the starting TF doesn't exist in the network, exit early
        if start_tf not in self.grn_df[start_col].values:
            return []

        full_graph_edges = []
        visited_tfs = set()

        # Queue contains TFs that we need to discover downstream targets for
        queue = deque([start_tf])
        visited_tfs.add(start_tf)

        while queue:
            current_tf = queue.popleft()

            # Find all direct targets of the current TF
            targets = (
                self.get_genes_from_tf(current_tf)
                if choose_target
                else self.get_tfs_from_gene(current_tf)
            )

            for edge in targets:
                target_gene = edge[target_col]
                regulation_type = edge[self.regulation]

                # Append the edge in a clean structural format
                full_graph_edges.append(
                    {
                        start_col: current_tf,
                        target_col: target_gene,
                        self.regulation: regulation_type,
                    }
                )

                # If the target gene is ALSO a TF itself and we haven't explored it yet,
                # add it to the queue to trace the next layer of the cascade
                if (
                    target_gene in self.grn_df[start_col].values
                    and target_gene not in visited_tfs
                ):
                    visited_tfs.add(target_gene)
                    queue.append(target_gene)

        return full_graph_edges


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
            "num_perturbs": 1,
            "gene_col_name": None,
            "plot_grns": True,
        }

    def _get_highly_variable_genes(self, train_dataset, n_top_genes):
        """Get the top n highly variable genes from the dataset."""
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

    def _get_gene_col(self, dataset):
        return (
            dataset.var[self.params["gene_col_name"]]
            if self.params["gene_col_name"]
            else dataset.var_names
        )

    def _submetric_eval(self, output_path, dataset: BaseDataset, method):
        """
        Run the submetric evaluation for GRN analyses.

        Args:
            output_path (str): The path to the output directory.
            dataset (Dataset): The dataset object.
            method (Method): The method object.
        """
        # set the random seed to ensure reproducibility
        np.random.seed(self.config.random_seed)

        self.grn = GRN(grn_path=self.params["grn_path"])

        # let's print out the overlap of genes
        train_dataset = load_train_dataset(output_path)
        genes_in_grn = self.grn.get_all_genes()
        genes_in_dataset = set(self._get_gene_col(train_dataset).tolist())

        overlapping_genes = sorted(list(genes_in_grn.intersection(genes_in_dataset)))
        logging.debug(f"Number of genes in GRN: {len(genes_in_grn)}")
        logging.debug(f"Number of genes in dataset: {len(genes_in_dataset)}")
        logging.debug(f"Number of overlapping genes: {len(overlapping_genes)}")
        logging.debug(f"Example overlapping genes: {overlapping_genes[:10]}")

        # genes in the dataset that aren't in the GRN
        non_overlapping_genes = sorted(list(genes_in_dataset.difference(genes_in_grn)))

        # Filter rows in .var where the gene_col values are in our overlapping list
        filtered_dataset = train_dataset[
            :,
            self._get_gene_col(train_dataset).isin(overlapping_genes),
        ].copy()

        # 3. Log the new dataset shape to verify
        logging.debug(f"Filtered dataset shape: {filtered_dataset.shape}")

        # Then we need to select the genes to perturb.
        # We either select the genes chosen in the parameters,
        # or select the top n highly variable genes in the dataset.
        genes = self.params["genes"]
        if genes is None or len(genes) == 0:
            genes = self._get_highly_variable_genes(
                filtered_dataset, self.params["num_genes"]
            )

        # next we handle each gene itself
        for gene in genes:
            self._handle_gene(
                gene,
                dataset.get_test_dataset_dir(),
                overlapping_genes,
                non_overlapping_genes,
                output_path,
            )

    def get_perturbation_genes(
        self,
        grn_genes,
        regulator_genes,
        overlapping_genes,
        non_overlapping_genes,
        is_target,
    ):
        """
        Create the 6 x n submetric configs for the perturbations we want to run for this gene.
        Where we do the random genes + genes that regulate/are regulated
        """
        # first let's choose n random genes from regulator genes
        chosen_regulator_genes = (
            np.random.choice(
                regulator_genes, size=self.params["num_perturbs"], replace=False
            )
            if len(regulator_genes) >= self.params["num_perturbs"]
            else regulator_genes
        )

        # get the tf/target based on what's available
        chosen_regulator_genes = [
            regulator_gene[
                self.grn.tf_col
                if self.grn.tf_col in regulator_gene
                else self.grn.gene_col
            ]
            for regulator_gene in chosen_regulator_genes
        ]

        # then let's choose n random genes that aren't in the GRN at all
        random_grn_genes = sorted(list(set(overlapping_genes) - set(grn_genes)))
        chosen_random_grn_genes = (
            np.random.choice(
                random_grn_genes, size=self.params["num_perturbs"], replace=False
            )
            if len(random_grn_genes) >= self.params["num_perturbs"]
            else random_grn_genes
        )

        chosen_random_non_grn_genes = (
            np.random.choice(
                non_overlapping_genes,
                size=self.params["num_perturbs"],
                replace=False,
            )
            if len(non_overlapping_genes) >= self.params["num_perturbs"]
            else non_overlapping_genes
        )

        logging.debug(
            f"Chosen regulator genes for perturbation: {chosen_regulator_genes}"
        )
        logging.debug(
            f"Chosen random GRN genes for perturbation: {chosen_random_grn_genes}"
        )
        logging.debug(
            f"Chosen random non-GRN genes for perturbation: {chosen_random_non_grn_genes}"
        )

        return {
            f'{"tf" if not is_target else "target"}_gene': chosen_regulator_genes,
            "random_grn_gene": chosen_random_grn_genes,
            "random_non_grn_gene": chosen_random_non_grn_genes,
        }

    def _handle_gene(
        self, gene, dataset_path, overlapping_genes, non_overlapping_genes, output_path
    ):
        """
        Given a certain gene, we find:
            - the genes that regulates it

            and then:
            1) a random gene that isn't related to the gene in the GRN
            2) a random gene that isn't related to the gene and is not in the GRN

            and repeat with
            - the genes that it regulates

        And we choose n genes from each category, and test both up and down regulating,
        and calculate its perturbation.

        In total we need to do: 2 (regulation direction) x 2 (up/down) x 3n (number of genes in each category) perturbations

        Because this is likely a lot, we probably should just do n = 1 for now, which is 12 perturbations per gene, which is quite a bit.
        """
        logging.debug(f"Gene: {gene}")
        # First we do the TF regulating the gene
        regulating_tfs = self.grn.get_tfs_from_gene(gene)
        full_tf_graph = self.grn.get_full_graph(gene, choose_target=False)
        logging.debug(f"Regulating TFs: {regulating_tfs}")
        logging.debug(
            f"Full TF Graph length: {len(self._genes_from_graph(full_tf_graph))}"
        )

        self._handle_tf_graphs(
            gene,
            regulating_tfs,
            full_tf_graph,
            overlapping_genes,
            non_overlapping_genes,
            dataset_path,
            output_path,
            is_target=False,
        )

        # Then we do the genes that are regulated by the gene
        regulated_genes = self.grn.get_genes_from_tf(gene)
        full_target_graph = self.grn.get_full_graph(gene, choose_target=True)
        logging.debug(f"Regulated Genes: {regulated_genes}")
        logging.debug(
            f"Full Target Graph length: {len(self._genes_from_graph(full_target_graph))}"
        )

        # TODO: handle this differently! we want to perturb gene, not regulated_genes
        self._handle_tf_graphs(
            gene,
            regulated_genes,
            full_target_graph,
            overlapping_genes,
            non_overlapping_genes,
            dataset_path,
            output_path,
            is_target=True,
        )

    def _handle_tf_graphs(
        self,
        gene,
        regulator_genes,
        full_graph,
        overlapping_genes,
        non_overlapping_genes,
        dataset_path,
        output_path,
        is_target,
    ):
        if len(full_graph) > 0 and self.params["plot_grns"]:
            self._plot_graph(full_graph, gene, dataset_path, is_target=is_target)

        if len(regulator_genes) == 0:
            return

        logging.debug(f"Gene: {gene}, {'TF' if not is_target else 'Target'} graph")
        perturbed_genes = self.get_perturbation_genes(
            self._genes_from_graph(full_graph),
            regulator_genes,
            overlapping_genes,
            non_overlapping_genes,
            is_target=is_target,
        )

        perturbation_results = {"upregulate": {}, "downregulate": {}}

        # now let's do the perturbations for each category of genes
        for category, genes_to_perturb in perturbed_genes.items():
            for perturbed_gene in genes_to_perturb:
                logging.debug(f"Perturbing {category} gene: {perturbed_gene}")
                perturbation_results["upregulate"][
                    f"{perturbed_gene}_{category}"
                ] = self._run_perturbation(gene, perturbed_gene, is_upregulate=True)
                perturbation_results["downregulate"][
                    f"{perturbed_gene}_{category}"
                ] = self._run_perturbation(gene, perturbed_gene, is_upregulate=False)

        # now let's plot both everything in upregulate and downregulate
        # where we plot the distribution for the gene -- let's collect all tps together though!
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        train_dataset = load_train_dataset(output_path)

        # finally add the baseline here, where we have the baseline expression
        # coming from the train dataset itself
        # also check if it's sparse, if so, we need to convert it to dense

        # we also need to make sure the lengths are the same,
        # so we will be taking the 1st to second last timepoint
        tps = sorted(
            list(train_dataset.obs[ObservationColumns.TIMEPOINT.value].unique())
        )
        filtered_dataset = train_dataset[
            train_dataset.obs[ObservationColumns.TIMEPOINT.value].isin(tps[:-1])
        ]
        baseline_gene_expr = filtered_dataset[
            :, self._get_gene_col(filtered_dataset).isin([gene])
        ].X
        if issparse(baseline_gene_expr):
            baseline_gene_expr = baseline_gene_expr.toarray()
        baseline_gene_expr = baseline_gene_expr.flatten()

        for regulation_direction, results in perturbation_results.items():
            plt.figure(figsize=(12, 8))
            plot_data = {}

            for perturbed_gene, (gene_expression, _) in results.items():
                # we will plot the distribution of expression across all timepoints for this gene
                all_expression_values = np.concatenate(
                    list(gene_expression.values())
                ).flatten()
                plot_data[perturbed_gene] = all_expression_values

            plot_data["baseline"] = baseline_gene_expr

            plot_df = pd.DataFrame(plot_data)

            sns.kdeplot(data=plot_df, fill=True, alpha=0.3, palette="Set2")

            plt.title(
                f"Distribution of expression values for {gene} when {regulation_direction} different genes"
            )
            plt.xlabel("Expression value")
            plt.ylabel("Estimated Density")
            dir_path = os.path.join(output_path, "gene_dist_plots", gene)
            os.makedirs(dir_path, exist_ok=True)
            plt.savefig(
                os.path.join(
                    dir_path, f"{regulation_direction}_perturbation_distribution.png"
                )
            )
            plt.close()
            exit()

    def _run_perturbation(self, gene, perturbed_gene, is_upregulate):
        """
        Run the perturbation for a given gene and label in both up/down regulation.
        """
        submetric_dict = {
            "name": "GlobalPerturbationGeneExpression",
            "affected_genes": [str(gene)],
            "perturbation_set_config": {
                "gene_col_name": self.params["gene_col_name"],
                "knockin_genes": [str(perturbed_gene)] if is_upregulate else [],
                "knockout_genes": [str(perturbed_gene)] if not is_upregulate else [],
            },
        }
        return self._run_submetric(submetric_dict)

    def _genes_from_graph(self, graph):
        """
        Given a list of edges, get the set of genes.
        """
        genes = set()
        for edge in graph:
            genes.add(edge[self.grn.tf_col])
            genes.add(edge[self.grn.gene_col])
        return sorted(list(genes))

    def _plot_graph(self, full_graph, gene, dataset_dir, is_target):
        """
        Given a full graph of the cascade, we plot it using networkx and save it to the output directory.
        """
        title = f"Full {'Target' if is_target else 'TF'} Graph for {gene}"

        G = nx.DiGraph()

        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        for edge in full_graph:
            G.add_edge(
                edge[self.grn.tf_col],
                edge[self.grn.gene_col],
                regulation=edge[self.grn.regulation],
            )

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        edge_colors = [
            "green"
            if G[u][v]["regulation"] == "Activation"
            else ("red" if G[u][v]["regulation"] == "Repression" else "gray")
            for u, v in G.edges()
        ]
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color=edge_colors,
            node_size=2000,
            font_size=10,
        )
        plt.title(title)
        dir_path = os.path.join(dataset_dir, "grn_plots", gene)
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(
            os.path.join(dir_path, f"grn_cascade_{'target' if is_target else 'tf'}.png")
        )
        plt.close()
