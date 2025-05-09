import numpy as np
import pytest
import torch
import tqdm

from route_choice.recursive_logit import RecursiveLogit


@pytest.mark.parametrize(
    "rl_tutorial_dataset,use_vi",
    [({"n_samples": 1000, "seed": 123}, True), ({"n_samples": 1000, "seed": 123}, False)],
    indirect=["rl_tutorial_dataset"],
)
def test_rl_tutorial_dataset(rl_tutorial_dataset, use_vi):
    batch, feat_scaler, n_feats = rl_tutorial_dataset

    model = RecursiveLogit(n_feats, link_constant=True, use_value_iteration=use_vi)
    optim = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, threshold=1e-4, threshold_mode="rel", patience=100, min_lr=1e-4
    )

    progress_bar = tqdm.trange(500)
    for epoch in progress_bar:
        model.train()
        optim.zero_grad()

        loss = model.train_step(batch, loss_reduction="sum")

        loss.backward()
        optim.step()
        scheduler.step(loss)

        progress_bar.set_postfix({"loss": loss.detach().item(), "lr": scheduler.get_last_lr()[0]})

    params = model.get_params()
    beta = params["beta"] / feat_scaler.scale_
    lc = params["link_constant"] - params["beta"] * feat_scaler.mean_ / feat_scaler.scale_
    assert np.isclose(beta, np.array([-2.0]), rtol=0.1), f"beta not close to expected value of -2 ({beta})"
    assert np.isclose(lc, np.array([-0.01]), atol=0.04), f"link constant not close to expected value of -0.01 ({lc})"
