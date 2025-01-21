import torch


class MultinomialLogit(torch.torch.nn.Module):
    def __init__(self, n_feats, n_alts, ref_alt=0):
        super().__init__()
        self.n_feats = n_feats  # number of features
        self.n_alts = n_alts  # number of alternatives

        self.coeffs = torch.nn.Linear(self.n_feats, 1, bias=False)
        self.biases = torch.nn.Parameter(torch.zeros(1, self.n_alts - 1))
        self.bias_mask = torch.ones(self.n_alts, dtype=bool)
        self.bias_mask[ref_alt] = False

    def forward(self, feats):
        # a bigger batch of features where each alternative is a sample, i.e. (batch_size * n_alts) x n_feats
        feats_flat = feats.flatten(0, 1)
        util_flat = self.coeffs(feats_flat)  # utils (batch_size * n_alts) x 1
        util = util_flat.reshape(-1, self.n_alts)  # batch size x n alts
        util[:, self.bias_mask] += self.biases  # bias (ASC) is a constant relative to the reference alternative
        return util

    def train_step(self, feats, labels):
        utils = self.forward(feats)
        loss = torch.nn.functional.cross_entropy(utils, labels, reduction="mean")
        return loss

    def get_params(self):
        params = dict(self.named_parameters())
        return {"asc": params["biases"].detach(), "beta": params["coeffs.weight"].detach()}
