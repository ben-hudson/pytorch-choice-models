import numpy as np
import pytest
import torch

from statsmodels.datasets import modechoice
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def mode_choice_dataset():
    mode_data = modechoice.load_pandas()
    mode_df = mode_data["data"]
    feat_cols = ["gc", "ttme", "hinc"]

    n_alts = len(mode_df["mode"].unique())
    n_feats = len(feat_cols)

    feats = []
    choices = []
    for _, feats_and_choices in mode_df.groupby("individual"):
        feats_and_choices = feats_and_choices.sort_values("mode")
        assert len(feats_and_choices) == n_alts, f"expected {n_alts} alternatives but got {len(feats_and_choices)}"
        feats.append(feats_and_choices[feat_cols].values)
        choices.append(feats_and_choices["choice"].values)

    feats_np = np.vstack(feats)
    labels_np = np.argmax(choices, axis=-1)

    feat_scaler = StandardScaler()
    feats_scaled_np = feat_scaler.fit_transform(feats_np)

    feats = torch.as_tensor(feats_scaled_np).reshape(-1, n_alts, n_feats).float()
    labels = torch.as_tensor(labels_np)
    # not every feature is taken into account for every alternative
    # for example, the hinc (household income) is only taken into account for mode 1
    feat_mask = torch.as_tensor(
        [
            [True, True, True],
            [True, True, False],
            [True, True, False],
            [True, True, False],
        ]
    )

    return feats, feat_mask, feat_scaler, labels, n_feats, n_alts
