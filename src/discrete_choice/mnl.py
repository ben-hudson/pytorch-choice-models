import torch


class MultinomialLogit(torch.torch.nn.Module):
    def __init__(self, n_feats: int, n_alts: int, ref_alt: int = 0):
        super().__init__()
        self.n_feats = n_feats  # number of features
        self.n_alts = n_alts  # number of alternatives

        # "beta" coefficients
        self.coeffs = torch.nn.Linear(self.n_feats, 1, bias=False)
        # alternative specific offsets relative to the reference alternative
        self.biases = torch.nn.Parameter(torch.zeros(1, self.n_alts - 1))
        self.bias_mask = torch.ones(self.n_alts, dtype=bool)
        self.bias_mask[ref_alt] = False

    # expects a 3D batch of features, i.e. batch size x n alts x n feats
    def forward(self, feats: torch.Tensor):
        # collapse into 2D batch of features, i.e. (batch_size * n_alts) x n_feats
        feats_flat = feats.flatten(0, 1)
        util_flat = self.coeffs(feats_flat)  # flat batch of deterministic util (batch_size * n_alts) x 1
        util = util_flat.reshape(-1, self.n_alts)  # 2D batch of deterministic util, i.e. batch size x n alts
        util[:, self.bias_mask] += self.biases  # bias (ASC) is relative to the reference alternative
        return util

    def train_step(self, feats: torch.Tensor, labels: torch.Tensor, loss_reduction: str = "mean"):
        utils = self.forward(feats)
        loss = torch.nn.functional.cross_entropy(utils, labels, reduction=loss_reduction)
        return loss

    def get_params(self):
        params = dict(self.named_parameters())
        return {"asc": params["biases"].detach(), "beta": params["coeffs.weight"].detach()}
