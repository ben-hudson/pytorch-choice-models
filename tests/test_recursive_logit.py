import numpy as np
import pytest
import torch
import tqdm

from torch_geometric.loader import DataLoader

from route_choice.recursive_logit import RecursiveLogit


@pytest.mark.parametrize(
    "rl_tutorial_dataset,use_vi",
    [({"n_samples": 1000, "seed": 321}, True), ({"n_samples": 1000, "seed": 321}, False)],
    indirect=["rl_tutorial_dataset"],
)
def test_rl_tutorial_dataset(rl_tutorial_dataset, use_vi):
    dataset, feat_scaler, n_feats = rl_tutorial_dataset
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = RecursiveLogit(n_feats, link_constant=True, use_value_iteration=use_vi)
    optim = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, threshold=1e-4, threshold_mode="rel", patience=50, min_lr=1e-4
    )

    progress_bar = tqdm.trange(200)
    for epoch in progress_bar:
        epoch_loss = 0
        for batch in loader:
            feats_scaled_np = feat_scaler.transform(batch.edge_attr.numpy())
            batch.feats = torch.as_tensor(feats_scaled_np, dtype=torch.float32)
            batch.choice = batch.path_edges

            model.train()
            optim.zero_grad()

            loss = model.train_step(batch, loss_reduction="sum")
            loss.backward()
            optim.step()

            epoch_loss += loss.detach()

        scheduler.step(epoch_loss)
        progress_bar.set_postfix({"loss": epoch_loss.item(), "lr": scheduler.get_last_lr()[0]})

    params = model.get_params()
    beta = params["beta"] / feat_scaler.scale_
    lc = params["link_constant"] - params["beta"] * feat_scaler.mean_ / feat_scaler.scale_
    assert np.isclose(beta, np.array([-2.0]), rtol=0.1), f"beta not close to expected value of -2 ({beta})"
    assert np.isclose(lc, np.array([-0.01]), atol=0.04), f"link constant not close to expected value of -0.01 ({lc})"
