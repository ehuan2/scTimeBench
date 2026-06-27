from scTimeBench.metrics.meta.base import MetaMetric
from scTimeBench.shared.utils import load_test_dataset
from scTimeBench.shared.constants import ObservationColumns
from scTimeBench.shared.dataset.base import BaseDataset
from scTimeBench.shared.perturbation_set import GlobalPerturbationSet
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
            "plot_grns": False,
            "use_hvgs": False,
            # by default don't filter by cell type, but if specified
            # we will filter by that cell type
            "cell_type": None,
        }

    def _cell_type_path(self):
        return "all" if self.params["cell_type"] is None else self.params["cell_type"]

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

    def _resolve_genes(self, filtered_dataset, dataset):
        """
        Selects which genes to use for perturbation analysis
        """
        genes = self.params["genes"]
        if genes is not None and len(genes) > 0:
            return genes

        if self.params["use_hvgs"]:
            return self._get_highly_variable_genes(
                filtered_dataset, self.params["num_genes"]
            )

        # collect the genes from perturbation lineages
        assert (
            dataset.cell_lineage_genes is not None
        ), "Dataset must have cell_lineage_genes defined for perturbation analysis."
        assert (
            dataset.cell_lineage_genes.get("gene_col_name")
            == self.params["gene_col_name"]
        ), "Gene column name mismatch between dataset and config."

        genes = []
        for transition in dataset.cell_lineage_genes["cell_lineage_genes"]:
            if (
                self.params["cell_type"] is not None
                and transition["start"] != self.params["cell_type"]
            ):
                continue
            for end_cell in transition["targets"]:
                genes.extend(end_cell["genes"])

        genes = list(set(genes))  # remove duplicates

        logging.debug(f"Genes from perturbation lineage: {genes}")

        if len(genes) == 0:
            raise ValueError(
                "No genes found from perturbation lineage. "
                "Please ensure that the GRN has enough genes, to use the use_hvgs flag "
                "or to specify genes in the config."
            )

        return genes

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
        test_dataset = load_test_dataset(output_path)
        genes_in_grn = self.grn.get_all_genes()
        genes_in_dataset = set(self._get_gene_col(test_dataset).tolist())

        overlapping_genes = sorted(list(genes_in_grn.intersection(genes_in_dataset)))
        logging.debug(f"Number of genes in GRN: {len(genes_in_grn)}")
        logging.debug(f"Number of genes in dataset: {len(genes_in_dataset)}")
        logging.debug(f"Number of overlapping genes: {len(overlapping_genes)}")
        logging.debug(f"Example overlapping genes: {overlapping_genes[:10]}")

        # genes in the dataset that aren't in the GRN
        non_overlapping_genes = sorted(list(genes_in_dataset.difference(genes_in_grn)))

        # Filter rows in .var where the gene_col values are in our overlapping list
        filtered_dataset = test_dataset[
            :,
            self._get_gene_col(test_dataset).isin(overlapping_genes),
        ].copy()

        # 3. Log the new dataset shape to verify
        logging.debug(f"Filtered dataset shape: {filtered_dataset.shape}")

        # Then we need to select the genes to perturb.
        # We either select the genes chosen in the parameters,
        # or select the top n highly variable genes in the dataset.
        genes = self._resolve_genes(filtered_dataset, dataset)

        # next we handle each gene itself
        for gene in genes:
            self._handle_gene(
                gene,
                dataset.get_test_dataset_dir(),
                overlapping_genes,
                non_overlapping_genes,
                output_path,
            )

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
            is_gene_tf=False,
        )

        # Then we do the gene being the TF
        regulated_genes = self.grn.get_genes_from_tf(gene)
        full_target_graph = self.grn.get_full_graph(gene, choose_target=True)
        logging.debug(f"Regulated Genes: {regulated_genes}")
        logging.debug(
            f"Full Target Graph length: {len(self._genes_from_graph(full_target_graph))}"
        )

        self._handle_tf_graphs(
            gene,
            regulated_genes,
            full_target_graph,
            overlapping_genes,
            non_overlapping_genes,
            dataset_path,
            output_path,
            is_gene_tf=True,
        )

    class GeneKeys:
        RANDOM_GRN_GENE = "random_grn"
        RANDOM_NON_GRN_GENE = "random_non_grn"
        TARGET_RANDOM_GENE = "target_random"
        TO_PERTURB = "tf"
        TARGET_GENE = "target"

    def _get_perturbation_genes(
        self,
        grn_genes,
        regulator_genes,
        overlapping_genes,
        non_overlapping_genes,
        is_gene_tf,
        gene,
    ):
        """
        Create the 6 x n submetric configs for the perturbations we want to run for this gene.
        Where we do the random genes + genes that regulate/are regulated

        Returns:
            dict (str -> List[str]):
                dictionary of different categories of genes to perturb
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
        # and if we're dealing with the gene being a TF
        # we need to make sure that the random gene is not in the TF graph of the target
        tf_graph_of_target = (
            self._genes_from_graph(self.grn.get_full_graph(gene, choose_target=False))
            if is_gene_tf
            else []
        )

        random_grn_genes = sorted(
            list(set(overlapping_genes) - set(grn_genes) - set(tf_graph_of_target))
        )
        chosen_random_grn_genes = (
            np.random.choice(
                random_grn_genes, size=self.params["num_perturbs"], replace=False
            )
            if len(random_grn_genes) >= self.params["num_perturbs"]
            else random_grn_genes
        )

        # get n random genes not in the dataset
        chosen_random_non_grn_genes = (
            np.random.choice(
                non_overlapping_genes,
                size=self.params["num_perturbs"],
                replace=False,
            )
            if len(non_overlapping_genes) >= self.params["num_perturbs"]
            else non_overlapping_genes
        )

        # take a gene that is not from the GRN but still in the dataset
        # that is not being perturbed
        left_random_grn_genes = sorted(
            list(set(random_grn_genes) - set(chosen_random_grn_genes))
        )
        to_measure_random_genes = (
            np.random.choice(
                left_random_grn_genes,
                size=self.params["num_perturbs"],
                replace=False,
            )
            if len(left_random_grn_genes) >= self.params["num_perturbs"]
            else left_random_grn_genes
        )

        random_genes = {
            self.GeneKeys.RANDOM_GRN_GENE: chosen_random_grn_genes,
            self.GeneKeys.RANDOM_NON_GRN_GENE: chosen_random_non_grn_genes,
            self.GeneKeys.TARGET_RANDOM_GENE: to_measure_random_genes,
        }

        return {
            # we perturb the gene itself if the gene is a TF
            # otherwise we perturb the regulator genes
            self.GeneKeys.TO_PERTURB: [gene] if is_gene_tf else chosen_regulator_genes,
            self.GeneKeys.TARGET_GENE: chosen_regulator_genes if is_gene_tf else [gene],
            **random_genes,
        }

    def _grn_path(self):
        return os.path.splitext(os.path.basename(self.params["grn_path"]))[0]

    def _handle_tf_graphs(
        self,
        gene,
        regulator_genes,
        full_graph,
        overlapping_genes,
        non_overlapping_genes,
        dataset_path,
        output_path,
        is_gene_tf,
    ):
        # this is a dictionary that is_gene_tf maps different keys to
        self.gene_tf_target_dict = {
            "graph_type": "Target" if is_gene_tf else "TF",
            "gene_type": "TF" if is_gene_tf else "Target",
        }

        if len(full_graph) > 0 and self.params["plot_grns"]:
            self._plot_graph(full_graph, gene, dataset_path)

        # if there is no regulator genes, we can't do any perturbations, skip
        if len(regulator_genes) == 0:
            return

        logging.debug(f"Gene: {gene}, {self.gene_tf_target_dict['graph_type']} graph")
        perturbed_genes = self._get_perturbation_genes(
            self._genes_from_graph(full_graph),
            regulator_genes,
            overlapping_genes,
            non_overlapping_genes,
            is_gene_tf=is_gene_tf,
            gene=gene,
        )

        perturbation_results = {"upregulate": {}, "downregulate": {}}

        target_genes = np.concatenate(
            (
                perturbed_genes[self.GeneKeys.TARGET_RANDOM_GENE],
                perturbed_genes[self.GeneKeys.TARGET_GENE],
            )
        )
        to_perturb_keys = [
            self.GeneKeys.TO_PERTURB,
            self.GeneKeys.RANDOM_GRN_GENE,
            self.GeneKeys.RANDOM_NON_GRN_GENE,
        ]

        logging.debug(
            f"Genes to perturb: {[(key, perturbed_genes[key]) for key in to_perturb_keys]}"
        )
        logging.debug(f"Target genes: {target_genes}")

        for key in to_perturb_keys:
            for gene_to_perturb in perturbed_genes[key]:
                logging.debug(
                    f"Perturbing gene: {gene_to_perturb} for target genes: {target_genes}"
                )
                perturbation_results_for_gene = self._run_perturbation(
                    target_genes=target_genes, perturbed_gene=gene_to_perturb
                )
                perturbation_results["upregulate"][
                    f"{key}_{gene_to_perturb}"
                ] = perturbation_results_for_gene["upregulate"]
                perturbation_results["downregulate"][
                    f"{key}_{gene_to_perturb}"
                ] = perturbation_results_for_gene["downregulate"]

        # now let's plot both everything in upregulate and downregulate
        # where we plot the distribution for the gene -- let's collect all tps together though!
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # finally add the baseline here, where we have the baseline expression
        test_dataset = load_test_dataset(output_path)
        # used to ensure that the preprocessing is the same
        global_perturb = GlobalPerturbationSet({"cell_type": self.params["cell_type"]})
        test_dataset = global_perturb.preprocess(test_dataset.copy())

        # we also need to make sure the lengths are the same,
        # so we will be taking the 1st to second last timepoint
        tps = sorted(
            list(test_dataset.obs[ObservationColumns.TIMEPOINT.value].unique())
        )
        filtered_dataset = test_dataset[
            test_dataset.obs[ObservationColumns.TIMEPOINT.value].isin(tps[:-1])
        ]

        # now let's do this so that we plot side by side all the target genes!
        num_cols = len(target_genes)
        num_rows = len(list(perturbation_results.keys()))

        if num_cols == 0 or num_rows == 0:
            logging.warning(
                f"No target genes or perturbation results to plot for gene: {gene}. Skipping plotting."
            )
            return

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(5 * num_cols, 4 * num_rows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for row, (regulation_direction, results) in enumerate(
            perturbation_results.items()
        ):
            for col, target_gene in enumerate(target_genes):
                ax = axes[row][col]

                plot_data = {}

                for perturbed_gene, (gene_expression, _) in results.items():
                    # we will plot the distribution of expression across all timepoints for this gene
                    all_expression_values = np.concatenate(
                        list(gene_expression[target_gene])
                    ).flatten()
                    plot_data[perturbed_gene] = all_expression_values

                baseline_gene_expr = filtered_dataset[
                    :, self._get_gene_col(filtered_dataset).isin([target_gene])
                ].X
                if issparse(baseline_gene_expr):
                    baseline_gene_expr = baseline_gene_expr.toarray()
                baseline_gene_expr = baseline_gene_expr.flatten()

                plot_data["baseline"] = baseline_gene_expr

                plot_df = pd.DataFrame(plot_data)

                sns.kdeplot(data=plot_df, fill=True, alpha=0.3, palette="Set2", ax=ax)

                is_random = (
                    target_gene in perturbed_genes[self.GeneKeys.TARGET_RANDOM_GENE]
                )

                ax.set_title(
                    f"{'Random' if is_random else ''} {target_gene} when {regulation_direction} for "
                    f"{self._cell_type_path()}"
                )
                ax.set_xlabel("Expression value")
                ax.set_ylabel(f"Estimated Density of {target_gene} expression")

        # make this grn specific, i.e. take the tsv file name and use its root
        dir_path = os.path.join(
            output_path,
            "gene_dist_plots",
            self._grn_path(),
            self._cell_type_path(),
            gene,
        )
        os.makedirs(dir_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                dir_path, f"gene_as_{self.gene_tf_target_dict['gene_type']}.png"
            )
        )
        plt.close()

    def _run_perturbation(self, target_genes, perturbed_gene):
        """
        Run the perturbation for a given gene and label in both up/down regulation.
        """

        def _run_single_perturbation(target_genes, perturbed_gene, is_upregulate):
            submetric_dict = {
                "name": "GlobalPerturbationGeneExpression",
                "affected_genes": [str(target_gene) for target_gene in target_genes],
                "perturbation_set_config": {
                    "gene_col_name": self.params["gene_col_name"],
                    "knockin_genes": [str(perturbed_gene)] if is_upregulate else [],
                    "knockout_genes": [str(perturbed_gene)]
                    if not is_upregulate
                    else [],
                    "cell_type": self.params["cell_type"],
                },
            }
            return self._run_submetric(submetric_dict)

        return {
            "upregulate": _run_single_perturbation(
                target_genes, perturbed_gene, is_upregulate=True
            ),
            "downregulate": _run_single_perturbation(
                target_genes, perturbed_gene, is_upregulate=False
            ),
        }

    def _genes_from_graph(self, graph):
        """
        Given a list of edges, get the set of genes.
        """
        genes = set()
        for edge in graph:
            genes.add(edge[self.grn.tf_col])
            genes.add(edge[self.grn.gene_col])
        return sorted(list(genes))

    def _plot_graph(self, full_graph, gene, dataset_dir):
        """
        Given a full graph of the cascade, we plot it using networkx and save it to the output directory.
        """
        title = f"Full {self.gene_tf_target_dict['graph_type']} Graph for {gene}"

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

        def get_edge_color(regulation):
            if isinstance(regulation, str):
                if regulation.lower() == "activation":
                    return "green"
                elif regulation.lower() == "repression":
                    return "red"
            # check if the type is numeric
            elif isinstance(regulation, (int, float)):
                if regulation > 0:
                    return "green"
                elif regulation < 0:
                    return "red"
            return "gray"  # default color for unknown regulation types

        edge_colors = [get_edge_color(G[u][v]["regulation"]) for u, v in G.edges()]

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
        dir_path = os.path.join(dataset_dir, "grn_plots", self._grn_path(), gene)
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(
            os.path.join(
                dir_path,
                f"grn_cascade_{self.gene_tf_target_dict['graph_type'].lower()}.png",
            )
        )
        plt.close()
