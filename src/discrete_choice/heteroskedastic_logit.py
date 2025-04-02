import torch

from .multinomial_logit import MultinomialLogit


class HeteroskedasticLogit(MultinomialLogit):
    def __init__(self, n_feats, n_alts, ref_alt=0, scale_ref_alt=0):
        super().__init__(n_feats, n_alts, ref_alt)

        self.bias_logvar = torch.nn.Parameter(torch.zeros(1, self.n_alts - 1))
        self.bias_logvar_mask = torch.ones(self.n_alts, dtype=bool)
        self.bias_logvar_mask[scale_ref_alt] = False

    def forward(self, feats: torch.Tensor, n_samples: int = 1):
        util = super().forward(feats)  # deterministic utility is calculated according to multinomial logit

        # sample random utility according to learned scales
        bias_scale = self.bias_logvar.exp().sqrt()
        random_util = torch.randn(n_samples, util.size(0), self.n_alts - 1) * bias_scale

        # expand deterministic utilities so we can add random samples to them
        util_samples = util.repeat(n_samples, 1, 1)
        util_samples[:, :, self.bias_logvar_mask] += random_util
        return util_samples

    # maximum simulated likelihood
    def train_step(self, feats: torch.Tensor, labels: torch.Tensor, loss_reduction: str = "mean", n_samples: int = 100):
        util = self.forward(feats, n_samples=n_samples)
        probs = torch.nn.functional.softmax(util, dim=-1)
        logprob = probs.mean(dim=0).log()
        loss = torch.nn.functional.nll_loss(logprob, labels, reduction=loss_reduction)
        return loss

    def get_params(self):
        params = super().get_params()
        params["bias_scale"] = self.bias_logvar.exp().sqrt()
        return params
