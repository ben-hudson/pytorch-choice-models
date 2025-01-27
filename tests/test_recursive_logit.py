import pytest
import torch
import tqdm

from route_choice.recursive_logit import RecursiveLogit
from fixtures.route_choice import route_choice_dataset, route_choice_graph


@pytest.mark.parametrize("route_choice_dataset", [(500, 123)], indirect=True)
def test_recursive_logit(route_choice_dataset):
    batch, feat_scaler, n_feats = route_choice_dataset

    model = RecursiveLogit(n_feats, link_constant=True, loss_reduction="sum")
    optim = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, threshold=0.01, threshold_mode="rel", patience=40, min_lr=1e-4
    )

    progress_bar = tqdm.trange(2000)
    for epoch in progress_bar:
        model.train()
        optim.zero_grad()

        loss = model.train_step(batch)

        loss.backward()
        optim.step()
        scheduler.step(loss)

        progress_bar.set_postfix({"loss": loss.detach().item(), "lr": scheduler.get_last_lr()[0]})

    params = model.get_params()
    assert True
