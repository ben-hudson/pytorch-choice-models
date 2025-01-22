import torch
import tqdm
import numpy as np

from discrete_choice.multinomial_logit import MultinomialLogit


def test_mnl(mode_choice_dataset):
    feats, feat_mask, feat_scaler, labels, n_feats, n_alts = mode_choice_dataset

    model = MultinomialLogit(n_feats, n_alts, ref_alt=3)
    optim = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, threshold=0.01, threshold_mode="rel", patience=40, min_lr=1e-4
    )

    progress_bar = tqdm.trange(1000)
    for epoch in progress_bar:
        model.train()
        optim.zero_grad()

        masked_feats = feats * feat_mask.expand_as(feats)
        loss = model.train_step(masked_feats, labels, loss_reduction="sum")

        loss.backward()
        optim.step()
        scheduler.step(loss)

        progress_bar.set_postfix({"loss": loss.detach().item(), "lr": scheduler.get_last_lr()[0]})

    params = model.get_params()

    # expected values estimated by PyLogit
    # https://github.com/timothyb0912/pylogit/blob/master/examples/notebooks/More%20Mixed%20Logit--Heteroskedasticity%20and%20Nesting.ipynb
    assert torch.isclose(
        loss, torch.tensor(199.128), rtol=0.001
    ), f"log-likelihood was not close to value estimated by PyLogit - see test_mnl.py for details"
    assert torch.isclose(
        params["asc"], torch.tensor([5.2074, 3.8690, 3.1632]), rtol=0.1
    ).all(), "ASCs were not close to values estimated by PyLogit - see test_mnl.py for details"
    assert np.isclose(
        params["beta"] / feat_scaler.scale_, np.array([-0.0155, -0.0961, 0.0133]), rtol=0.1
    ).all(), "betas were not close to values estimated by PyLogit - see test_mnl.py for details"
