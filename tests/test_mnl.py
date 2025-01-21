import torch
from discrete_choice.mnl import MultinomialLogit


if __name__ == "__main__":
    torch.manual_seed(321)

    feats_train, labels_train, feats_val, labels_val, feat_mask, n_feats, n_alts = get_dataset()

    model = MultinomialLogit(n_feats, n_alts, ref_alt=3)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, threshold=0.001, threshold_mode="rel", patience=100)

    progress_bar = tqdm.trange(10000)
    for epoch in progress_bar:
        model.train()
        optim.zero_grad()

        masked_feats = feats_train * feat_mask.expand_as(feats_train)
        loss = model.train_step(masked_feats, labels_train, n_samples=1000)

        loss.backward()
        optim.step()

        with torch.no_grad():
            model.eval()
            masked_feats = feats_val * feat_mask.expand_as(feats_val)
            val_loss = model.train_step(masked_feats, labels_val, n_samples=1000)

            # plot the variance of the error component
            # if epoch % 100 == 0:
            #     samples = model.decoder(torch.randn(1000, model.n_alts))
            #     fig = plot_util_dist(samples)
            #     fig.savefig(f"error_component_epoch={epoch}.png")

        scheduler.step(val_loss)

        progress_bar.set_postfix(
            {"loss": loss.detach().item(), "val_loss": val_loss.item(), "lr": scheduler.get_last_lr()}
        )

    params = dict(model.named_parameters())
    print(params)
