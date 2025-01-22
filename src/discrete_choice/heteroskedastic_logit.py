import torch

from .multinomial_logit import MultinomialLogit


class HeteroskedasticLogit(MultinomialLogit):
    def __init__(self, n_feats, n_alts, ref_alt=0, scale_ref_alt=0):
        super().__init__(n_feats, n_alts, ref_alt)

        self.bias_logvar = torch.nn.Parameter(torch.zeros(1, self.n_alts - 1))
        self.bias_logvar_mask = torch.ones(self.n_alts, dtype=bool)
        self.bias_logvar_mask[scale_ref_alt] = False

    def forward(self, feats):
        util = super().forward(feats)  # deterministic utility is calculated according to multinomial logit

        # add random utility samples according to learned scales
        scale = self.bias_logvar.exp().sqrt()
        random_util = torch.randn(util.size(0), self.n_alts - 1) * scale
        util[:, self.bias_logvar_mask] += random_util
        return util

    # maximum simulated likelihood
    def train_step(self, feats: torch.Tensor, labels: torch.Tensor, loss_reduction: str = "mean", n_samples: int = 100):
        probs = []
        for _ in range(n_samples):
            # each call to forward is draws one sample
            util = self.forward(feats)
            prob = torch.nn.functional.softmax(util, dim=-1)
            probs.append(prob)
        logprob = torch.stack(probs).mean(dim=0).log()

        loss = torch.nn.functional.nll_loss(logprob, labels, reduction=loss_reduction)
        return loss
