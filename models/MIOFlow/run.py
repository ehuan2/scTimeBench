"""
MIOFlow runner script.

This script trains and evaluates MIOFlow on an AnnData dataset.
It keeps the BaseModel runner structure used across the project.
"""

# let's add the model_utils path
import os
from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA

from crispy_fishstick.model_utils.model_runner import main, BaseModel
from crispy_fishstick.shared.constants import ObservationColumns, RequiredOutputColumns

from MIOFlow.utils import set_seeds, config_criterion
from MIOFlow.models import make_model, Autoencoder
from MIOFlow.train import training_regimen, train_ae
from MIOFlow.geo import setup_distance
from MIOFlow.eval import generate_points


# next we want to prepare the data for training
def _to_dense(x):
    return x.toarray() if sp.issparse(x) else x


def prepare_data(ann_data, pca_dims: int, seed: int):
    """
    Prepare data for MIOFlow training with PCA embeddings.
    """
    # get the time points
    cell_tps = ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
    unique_tps = sorted(np.unique(cell_tps))

    # Fit PCA on all available training data in this runner
    X = _to_dense(ann_data.X)
    pca = PCA(n_components=pca_dims, svd_solver="arpack", random_state=seed)
    X_pca = pca.fit_transform(X)

    # Map timepoints to contiguous indices (required by MIOFlow)
    tp_to_idx: Dict = {tp: i for i, tp in enumerate(unique_tps)}

    df = pd.DataFrame(X_pca, columns=[f"d{i}" for i in range(1, pca_dims + 1)])
    df.insert(0, "samples", [tp_to_idx[tp] for tp in cell_tps])

    return df, unique_tps, tp_to_idx, pca


def model_setup(model_features: int, layers: List[int], n_timepoints: int, use_cuda: bool):
    # Model setup
    sde_scales = [0.2] * n_timepoints
    return make_model(
        model_features,
        layers,
        activation="CELU",
        scales=sde_scales,
        use_cuda=use_cuda,
    )


def model_training(
    df_train: pd.DataFrame,
    groups: List[int],
    metadata: Dict,
    model_features: int,
    use_cuda: bool,
):
    # Model training
    n_local_epochs = metadata.get("n_local_epochs", 20)
    n_epochs = metadata.get("n_epochs", 80)
    n_post_local_epochs = metadata.get("n_post_local_epochs", 0)
    batch_size = metadata.get("batch_size", 64)
    layers = [int(x) for x in metadata.get("layers", "16,32,16").split(",") if x.strip() != ""]
    lambda_density = metadata.get("lambda_density", 5.0)
    use_density_loss = metadata.get("use_density_loss", False)

    model = model_setup(model_features, layers, len(groups), use_cuda)
    criterion = config_criterion("ot")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    local_losses = defaultdict(list)
    ot_lambda_global = {i: 1.0 for i in range(len(groups))}

    training_regimen(
        n_local_epochs=n_local_epochs,
        n_epochs=n_epochs,
        n_post_local_epochs=n_post_local_epochs,
        exp_dir=metadata.get("output_path", "./Output/AnnData"),
        model=model,
        df=df_train,
        groups=groups,
        optimizer=optimizer,
        criterion=criterion,
        use_cuda=use_cuda,
        hold_one_out=False,
        hold_out="random",
        use_density_loss=use_density_loss,
        lambda_density=lambda_density,
        autoencoder=metadata.get("autoencoder", None),
        use_emb=False,
        use_gae=metadata.get("use_gae", False),
        sample_size=(batch_size,),
        reverse_schema=True,
        reverse_n=2,
        plot_every=100,
        n_points=100,
        n_trajectories=100,
        n_bins=100,
        local_losses=local_losses,
        batch_losses=[],
        globe_losses=[],
        ot_lambda_global=ot_lambda_global,
    )

    return model


class MIOFlow(BaseModel):
    def train(self, ann_data):
        """
        Training logic for MIOFlow.
        """
        cache_path = os.path.join(
            self.config["output_path"], "trained_mioflow_model.pth"
        )

        metadata = self.config.get("model", {}).get("metadata", {})
        pca_dims = metadata.get("pca_dims", 50)
        seed = metadata.get("seed", 0)
        use_gae = metadata.get("use_gae", False)
        gae_embedded_dim = metadata.get("gae_embedded_dim", 32)
        gae_encoder_layers = metadata.get("gae_encoder_layers", "")
        n_epochs_gae = metadata.get("n_epochs_gae", 10)
        gae_sample_size = metadata.get("gae_sample_size", 30)
        distance_type = metadata.get("distance_type", "gaussian")
        rbf_length_scale = metadata.get("rbf_length_scale", 0.5)
        knn = metadata.get("knn", 5)
        t_max = metadata.get("t_max", 5)

        set_seeds(seed)
        use_cuda = torch.cuda.is_available()

        if os.path.exists(cache_path):
            print("Trained MIOFlow model found, loading from file.")
            cache = torch.load(cache_path, map_location="cpu")
            self.pca = cache["pca"]
            self.tp_to_idx = cache["tp_to_idx"]
            self.idx_to_tp = cache["idx_to_tp"]
            model_features = cache["model_features"]
            layers = cache["layers"]
            self.model = model_setup(
                model_features, layers, len(self.tp_to_idx), use_cuda
            )
            self.model.load_state_dict(cache["model_state"])
            self.use_gae = cache.get("use_gae", False)
            self.autoencoder = None
            if self.use_gae:
                encoder_layers = cache["gae_encoder_layers"]
                gae = Autoencoder(
                    encoder_layers=encoder_layers,
                    decoder_layers=encoder_layers[::-1],
                    activation="ReLU",
                    use_cuda=use_cuda,
                )
                gae.load_state_dict(cache["autoencoder_state"])
                self.autoencoder = gae
            self.df_train = None
            return

        df_train, unique_tps, tp_to_idx, pca = prepare_data(
            ann_data,
            pca_dims=pca_dims,
            seed=seed,
        )
        self.df_train = df_train
        self.pca = pca
        self.tp_to_idx = tp_to_idx
        self.idx_to_tp = {i: tp for tp, i in tp_to_idx.items()}

        autoencoder = None
        if use_gae:
            if gae_encoder_layers.strip() == "":
                encoder_layers = [pca_dims, 8, gae_embedded_dim]
            else:
                encoder_layers = [
                    int(x) for x in gae_encoder_layers.split(",") if x.strip() != ""
                ]
                if encoder_layers[0] != pca_dims:
                    raise ValueError("First GAE encoder layer must match pca_dims.")
                if encoder_layers[-1] != gae_embedded_dim:
                    raise ValueError("Last GAE encoder layer must match gae_embedded_dim.")

            dist = setup_distance(
                distance_type,
                rbf_length_scale=rbf_length_scale,
                knn=knn,
                t_max=t_max,
            )

            gae = Autoencoder(
                encoder_layers=encoder_layers,
                decoder_layers=encoder_layers[::-1],
                activation="ReLU",
                use_cuda=use_cuda,
            )
            gae_optimizer = torch.optim.AdamW(gae.parameters())
            train_ae(
                gae,
                df_train,
                groups=sorted(df_train.samples.unique()),
                optimizer=gae_optimizer,
                n_epochs=n_epochs_gae,
                sample_size=(gae_sample_size,),
                noise_min_scale=0.09,
                noise_max_scale=0.15,
                dist=dist,
                recon=True,
                use_cuda=use_cuda,
                hold_one_out=False,
                hold_out="random",
            )
            autoencoder = gae
            gae_encoder_layers_cache = encoder_layers
        else:
            gae_encoder_layers_cache = None

        model_features = gae_embedded_dim if use_gae else pca_dims
        layers = [int(x) for x in metadata.get("layers", "16,32,16").split(",") if x.strip() != ""]

        metadata["output_path"] = self.config["output_path"]
        metadata["autoencoder"] = autoencoder
        metadata["use_gae"] = use_gae
        self.model = model_training(
            df_train,
            groups=sorted(df_train.samples.unique()),
            metadata=metadata,
            model_features=model_features,
            use_cuda=use_cuda,
        )
        self.autoencoder = autoencoder
        self.use_gae = use_gae

        cache_payload = {
            "pca": self.pca,
            "tp_to_idx": self.tp_to_idx,
            "idx_to_tp": self.idx_to_tp,
            "model_state": self.model.state_dict(),
            "model_features": model_features,
            "layers": layers,
            "use_gae": self.use_gae,
        }
        if self.use_gae and self.autoencoder is not None:
            cache_payload["autoencoder_state"] = self.autoencoder.state_dict()
            cache_payload["gae_encoder_layers"] = gae_encoder_layers_cache

        torch.save(cache_payload, cache_path)

    def generate(self, test_ann_data, expected_output_path):
        """
        Generation logic with interpolation for MIOFlow.
        Returns an AnnData object containing the generated samples.
        """
        self.model.eval()

        final_ann_data = test_ann_data.copy()

        # Generate embeddings for all timepoints using trained model
        all_tp_indices = [self.tp_to_idx[tp] for tp in sorted(self.tp_to_idx.keys())]
        cell_tps = test_ann_data.obs[ObservationColumns.TIMEPOINT.value].to_numpy()
        unique_tps = sorted(np.unique(cell_tps))
        n_sim_cells = test_ann_data[test_ann_data.obs[ObservationColumns.TIMEPOINT.value] == unique_tps[0]].shape[0]

        if self.df_train is None:
            # Rebuild a minimal df for generation when model loaded from cache
            X = self.pca.transform(_to_dense(test_ann_data.X))
            df_tmp = pd.DataFrame(
                X, columns=[f"d{i}" for i in range(1, X.shape[1] + 1)]
            )
            df_tmp.insert(
                0,
                "samples",
                [self.tp_to_idx[tp] for tp in cell_tps],
            )
            df_for_gen = df_tmp
        else:
            df_for_gen = self.df_train

        generated = generate_points(
            self.model,
            df_for_gen,
            n_points=n_sim_cells,
            sample_with_replacement=True,
            use_cuda=torch.cuda.is_available(),
            samples_key="samples",
            sample_time=all_tp_indices,
            autoencoder=self.autoencoder,
            recon=self.use_gae,
        )

        # generated shape: (n_timepoints, n_cells, n_pca)
        # Use the first timepoint embeddings as a stand-in for per-cell embeddings
        embeds = generated[0]

        if RequiredOutputColumns.EMBEDDING in self.required_outputs:
            final_ann_data.obsm[RequiredOutputColumns.EMBEDDING.value] = embeds

        if RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING in self.required_outputs:
            # Provide next timepoint embeddings for each cell by picking the next timepoint in order
            if generated.shape[0] > 1:
                next_embeds = generated[1]
            else:
                next_embeds = np.full_like(embeds, np.nan)
            final_ann_data.obsm[
                RequiredOutputColumns.NEXT_TIMEPOINT_EMBEDDING.value
            ] = next_embeds

        final_ann_data.write_h5ad(expected_output_path)


if __name__ == "__main__":
    main(MIOFlow)
