import numpy as np
import torch
import tqdm

from discrete_choice.heteroskedastic_logit import HeteroskedasticLogit


def test_heteroskedastic_logit(mode_choice_dataset):
    feats, feat_mask, feat_scaler, labels, n_feats, n_alts = mode_choice_dataset

    model = HeteroskedasticLogit(n_feats, n_alts, ref_alt=3, scale_ref_alt=2)
    # initialize the model with the values from the MNL model
    # they do this when estimating the model in PyLogit so I think it is fair
    # futhermore, it makes running the test much faster
    with torch.no_grad():
        model.coeffs.weight.data = torch.tensor([[-0.7432, -2.3961, 0.2615]])
        model.biases.data = torch.tensor([5.2074, 3.8690, 3.1632])

    optim = torch.optim.Adam(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, threshold=0.01, threshold_mode="rel", patience=10, min_lr=1e-4
    )

    progress_bar = tqdm.trange(100)
    for epoch in progress_bar:
        model.train()
        optim.zero_grad()

        masked_feats = feats * feat_mask.expand_as(feats)
        loss = model.train_step(masked_feats, labels, loss_reduction="sum", n_samples=1000)

        loss.backward()
        optim.step()
        scheduler.step(loss)

        progress_bar.set_postfix({"loss": loss.detach().item(), "lr": scheduler.get_last_lr()[0]})

    params = model.get_params()

    # expected values estimated by PyLogit
    # https://github.com/timothyb0912/pylogit/blob/master/examples/notebooks/More%20Mixed%20Logit--Heteroskedasticity%20and%20Nesting.ipynb
    assert torch.isclose(
        loss, torch.tensor(195.7015), rtol=0.01
    ), f"log-likelihood was not close to value estimated by PyLogit - see {__file__} for details"
    # with the exception of the betas, the confidence intervals are so wide on the parameters that it doesn't make sense to check them
    assert np.isclose(
        params["beta"] / feat_scaler.scale_, np.array([-0.0333, -0.1156, 0.0381]), rtol=0.2
    ).all(), f"betas were not close to values estimated by PyLogit - see {__file__} for details"
