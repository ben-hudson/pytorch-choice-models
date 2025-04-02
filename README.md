# Choice models in PyTorch
Discrete choice models explain how people pick one option from a set of distinct possibilities. They assume each option has certain features that individuals weigh against personal preferences and any unobserved influences when making a choice. Route choice models apply this framework to transportation, where the "alternatives" are possible routes from one location to another.

This repo contains PyTorch implementations for common discrete and route choice models.

## Discrete choice models
- Multinomial logit (MNL)
- Heteroskedastic logit

## Route choice models
- Recursive logit

# Other libraries
The purpose of this library is to provide discrete choice models that can be fitted using PyTorch optimizers.

If you're looking for a discrete choice Python library in general, [PyLogit](https://github.com/timothyb0912/pylogit) implements many models with [examples](https://github.com/timothyb0912/pylogit/tree/master/examples/notebooks)!
