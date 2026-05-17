"""
This module contains the Plotting class, which is responsible for all plotting functions related to scTimeBench results.
"""
from scTimeBench.config import Config
import os


class Plotting:
    """
    Class responsible for all plotting functions related to scTimeBench results, including:
    - Graph similarity heatmaps and scatter plots
    """

    def __init__(self, config: Config):
        self.config = config
        os.makedirs(self.config.plot_output_dir, exist_ok=True)

    def plot_graph_sim_from_csv(self, csv_path):
        """
        Plots graph similarity metrics from a CSV file.
        """
        self.plot_graph_sim_heatmap(csv_path)
        self.plot_graph_sim_scatter(csv_path, time_type="Real Time")
        self.plot_graph_sim_scatter(csv_path, time_type="Pseudotime")

    def plot_graph_sim_heatmap(self, csv_path):
        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        import logging

        # first turn off matplotlib debugging
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # --- CONFIGURATION ---
        INPUT_FILE = csv_path
        OUTPUT_FILE = self.config.plot_output_dir + "/log_fold_change_heatmap.svg"
        METRICS_OF_INTEREST = [
            "AUC_PRC",
            "AUC_ROC",
            "JaccardSimilarity",
        ]  # Adjust as needed

        # 1. Load and Rename
        df = pd.read_csv(INPUT_FILE)

        def rename_3x_groups(row):
            if "3x" in str(row["dataset"]):
                return f"{row['method']}-3x"
            return row["method"]

        df["method"] = df.apply(rename_3x_groups, axis=1)

        # 2. Pivot & Calculate LFC
        pivot_df = df.pivot_table(
            index=["dataset", "step_setting", "metric", "method", "threshold_type"],
            columns="time_type",
            values="result",
        ).reset_index()

        pivot_df["LFC"] = np.log2(
            pivot_df["Pseudotime"] / pivot_df["Real Time"] + np.finfo(float).eps
        )  # Add small value to avoid log(0)

        # 3. Filter Logic
        pivot_df = pivot_df[
            (pivot_df["metric"].isin(METRICS_OF_INTEREST))
            & (pivot_df["threshold_type"] == "prc")
        ]

        # 4. Get unique step_settings to determine number of subplots
        # ... (previous code)

        # 4. Get unique step_settings
        step_settings = pivot_df["step_setting"].unique()
        _, axes = plt.subplots(
            len(step_settings), 1, figsize=(14, 8 * len(step_settings))
        )

        if len(step_settings) == 1:
            axes = [axes]

        # STEEPER GRADIENT LOGIC:
        # We cap the visual range at 1.5. This means anything > 1.5x (or < -1.5x)
        # is fully saturated, making the "middle" transitions much sharper.
        clean_vals = pivot_df.replace([np.inf, -np.inf], np.nan)
        actual_max = clean_vals["LFC"].abs().max()
        max_abs = actual_max

        # 5. Plotting Loop
        for i, setting in enumerate(step_settings):
            ax = axes[i]
            # Filter data for this specific setting
            setting_df = pivot_df[pivot_df["step_setting"] == setting]

            # --- SWAPPED INDEX AND COLUMNS HERE ---
            # Rows (index) are now methods
            # Columns are now the multi-index of dataset and metric
            subplot_pivot = setting_df.pivot_table(
                index="method", columns=["dataset", "metric"], values="LFC"
            )

            # Set grey background for NaNs
            ax.set_facecolor("#E0E0E0")

            from matplotlib.colors import TwoSlopeNorm

            # Recalculate norm for this specific plot or use global
            norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

            # Draw Heatmap
            sns.heatmap(
                subplot_pivot,
                annot=False,
                cmap="seismic",  # Or "RdBu_r" based on your preference
                norm=norm,
                square=True,
                linewidths=0.5,
                cbar_kws={"label": "Log2 Fold Change"} if i == 0 else None,
                ax=ax,
            )

            # 6. MANUALLY DRAW SLASHES (Adjusted for new shape)
            rows, cols = subplot_pivot.shape
            for y in range(rows):
                for x in range(cols):
                    val = subplot_pivot.iloc[y, x]
                    if pd.isna(val) or np.isinf(val):
                        ax.text(
                            x + 0.5,
                            y + 0.5,
                            "/",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=14,
                            fontweight="bold",
                        )

            ax.set_title(f"Step Setting: {setting}", fontsize=16, fontweight="bold")
            ax.set_xlabel("Dataset / Metric")  # Changed from ylabel
            ax.set_ylabel("Method")  # Changed from "Dataset / Metric"

        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, format="svg")

    def plot_graph_sim_scatter(self, csv_path, time_type):
        """
        Plots a scatter plot of graph similarity metrics from a CSV file, with custom colors and legend.
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.colors as mcolors
        from matplotlib.lines import Line2D

        # Load data
        df = pd.read_csv(csv_path)

        # 1. Filter Data
        metrics_to_plot = ["JaccardSimilarity", "AUC_PRC", "AUC_ROC"]
        plot_df = df[
            (df["threshold_type"] == "prc")
            & (df["time_type"] == time_type)
            & (df["metric"].isin(metrics_to_plot))
        ].copy()

        # Sort and prepare combined labels
        plot_df["metric"] = pd.Categorical(
            plot_df["metric"], categories=metrics_to_plot, ordered=True
        )
        plot_df = plot_df.sort_values(["metric", "dataset"])

        dataset_order = list(dict.fromkeys(plot_df["dataset"]))

        # 2. Setup the Unified Palette and Explicit Order
        unique_methods = sorted(
            plot_df["method"].dropna().unique().tolist(), key=str.casefold
        )

        def find_method_name(methods, target_name):
            for method_name in methods:
                if method_name.casefold() == target_name.casefold():
                    return method_name
            return None

        moscot_method = find_method_name(unique_methods, "Moscot")
        wot_method = find_method_name(unique_methods, "WOT")
        correlation_method = find_method_name(unique_methods, "Correlation")
        ot_cfm_method = find_method_name(unique_methods, "ot_cfm")

        methods_to_move_last = [
            method_name
            for method_name in [
                moscot_method,
                wot_method,
                correlation_method,
                ot_cfm_method,
            ]
            if method_name is not None
        ]
        base_methods = [m for m in unique_methods if m not in methods_to_move_last]
        ordered_methods = base_methods + methods_to_move_last

        # Skip the original amber-like slot in Set1; reserve amber for Moscot explicitly.
        set1_candidates = sns.color_palette("Set1", len(base_methods) + 3)
        base_palette = [color for idx, color in enumerate(set1_candidates) if idx != 5]
        if len(base_palette) < len(base_methods):
            base_palette = sns.color_palette("tab20", len(base_methods))

        custom_palette = {}
        for idx, method_name in enumerate(base_methods):
            custom_palette[method_name] = base_palette[idx]

        if moscot_method is not None:
            custom_palette[moscot_method] = "#FFB000"  # amber
        if wot_method is not None:
            custom_palette[wot_method] = "#455A64"  # strong blue
        if ot_cfm_method is not None:
            custom_palette[ot_cfm_method] = "#5DACE5"  # olive
        if correlation_method is not None:
            custom_palette[correlation_method] = "#CC79A7"  # magenta-purple

        all_methods = ordered_methods
        plot_df["method"] = pd.Categorical(
            plot_df["method"], categories=all_methods, ordered=True
        )

        # 3. Setup the Grid
        g = sns.FacetGrid(
            plot_df,
            row="step_setting",
            col="metric",
            col_order=metrics_to_plot,
            height=3.8,
            aspect=0.78,
            margin_titles=True,
            sharex=True,
            sharey=True,
        )

        # 4. Define Plotting Function
        def layered_swarm(data, **kwargs):
            ax = plt.gca()

            # Now we just plot everything in one go using the unified palette
            sns.swarmplot(
                data=data,
                x="dataset",
                y="result",
                hue="method",
                hue_order=all_methods,
                order=dataset_order,
                palette=custom_palette,
                size=7,
                dodge=False,
                ax=ax,
                alpha=1.0,
            )

            # Keep seaborn's jitter placement, then replace Correlation circles with diamonds.
            if correlation_method is not None:
                correlation_color = np.array(
                    mcolors.to_rgba(custom_palette[correlation_method])
                )
                for collection in ax.collections:
                    facecolors = collection.get_facecolors()
                    if facecolors is None or len(facecolors) == 0:
                        continue

                    # seaborn puts multiple method colors in the same PathCollection,
                    # so we mask per point (not per collection).
                    rgb_close = np.isclose(
                        facecolors[:, :3], correlation_color[:3], atol=1e-3
                    )
                    corr_mask = np.all(rgb_close, axis=1)
                    if not np.any(corr_mask):
                        continue

                    offsets = collection.get_offsets()
                    if offsets is None or len(offsets) == 0:
                        continue

                    corr_offsets = offsets[corr_mask]
                    ax.scatter(
                        corr_offsets[:, 0],
                        corr_offsets[:, 1],
                        marker="D",
                        s=40,
                        c=[custom_palette[correlation_method]],
                        edgecolors="black",
                        linewidths=0.6,
                        zorder=collection.get_zorder() + 0.1,
                    )

                    updated_facecolors = facecolors.copy()
                    updated_facecolors[corr_mask, 3] = 0.0
                    collection.set_facecolors(updated_facecolors)

                    edgecolors = collection.get_edgecolors()
                    if edgecolors is not None and len(edgecolors) > 0:
                        if len(edgecolors) == 1 and len(updated_facecolors) > 1:
                            edgecolors = np.repeat(
                                edgecolors, len(updated_facecolors), axis=0
                            )
                        if len(edgecolors) == len(updated_facecolors):
                            updated_edgecolors = edgecolors.copy()
                            updated_edgecolors[corr_mask, 3] = 0.0
                            collection.set_edgecolors(updated_edgecolors)

            if ax.get_legend():
                ax.get_legend().remove()

        # 5. Map the function
        g.map_dataframe(layered_swarm)

        # 6. Final Polish
        g.set(ylim=(0, 1.05))
        g.set_axis_labels("Dataset", "Score (0.0 - 1.0)")

        for row_idx, axes_row in enumerate(g.axes):
            for col_idx, ax in enumerate(axes_row):
                ax.set_xticks(range(len(dataset_order)))
                ax.set_xticklabels(dataset_order, rotation=45, ha="right", fontsize=7)
                ax.set_xlim(-0.5, len(dataset_order) - 0.5)
                ax.tick_params(axis="y", labelsize=7, labelleft=True, left=True)
                ax.yaxis.set_label_position("left")
                ax.yaxis.tick_left()
                ax.set_ylabel(f"{metrics_to_plot[col_idx]} (0.0 - 1.0)", fontsize=8)

        # 7. Custom Legend
        legend_elements = []

        # Iterate through all methods and use the custom_palette dictionary for the colors
        for m in all_methods:
            is_correlation = correlation_method is not None and m == correlation_method
            marker = "D" if is_correlation else "o"
            marker_edge_color = "black" if is_correlation else "none"
            marker_edge_width = 0.6 if is_correlation else 0.0
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    label=m,
                    markerfacecolor=custom_palette[m],
                    markersize=6,
                    markeredgecolor=marker_edge_color,
                    markeredgewidth=marker_edge_width,
                )
            )

        # Calculate the number of columns needed to split the items evenly into 2 rows
        num_columns = (len(all_methods) + 1) // 2

        g.fig.legend(
            handles=legend_elements,
            loc="upper center",  # Anchor point on the legend box
            bbox_to_anchor=(
                0.5,
                0.0,
            ),  # Position relative to the figure (x=center, y=bottom)
            title="Methods",
            fontsize="small",
            ncol=num_columns,  # Automatically splits into 2 rows
        )

        plt.subplots_adjust(top=0.9, right=0.98, bottom=0.25, hspace=0.4, wspace=0.32)
        plt.savefig(
            f"{self.config.plot_output_dir}/graph_sim_scatter_{'real' if time_type == 'Real Time' else 'pseudotime'}.svg",
            format="svg",
            bbox_inches="tight",
        )
