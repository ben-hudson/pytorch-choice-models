# Choice models in PyTorch
Discrete choice models explain how people pick one option from a set of distinct possibilities. They assume each option has certain features that individuals weigh against personal preferences and any unobserved influences when making a choice. Route choice models apply this framework to transportation, where the "alternatives" are possible routes from one location to another.

This repo contains PyTorch implementations for some discrete and route choice models.

## Installation
Run `pip install .` in the root directory. If you want to edit the source code add the `-e` flag after `install`. Next, run `pytest -v tests/` to verify the installation.

## Discrete choice models
- [`MultinomialLogit`](https://github.com/ben-hudson/pytorch-choice-models/blob/main/src/discrete_choice/multinomial_logit.py) implements the standard MNL model.
- [`HeteroskedasticLogit`](https://github.com/ben-hudson/pytorch-choice-models/blob/main/src/discrete_choice/heteroskedastic_logit.py) implements a logit model with heteroskedastic error components (i.e. alternative specific errors).

### Examples
These tests fit the models to a mode choice dataset containing 210 individuals' preferences for traveling by plane, car, bus, or train between Sydney, Canberra, and Melbourne.
- [test_mnl.py](https://github.com/ben-hudson/pytorch-choice-models/blob/main/tests/test_mnl.py)
- [test_heteroskedastic_logit.py](https://github.com/ben-hudson/pytorch-choice-models/blob/main/tests/test_heteroskedastic_logit.py)

## Route choice models
- [`RecursiveLogit`](https://github.com/ben-hudson/pytorch-choice-models/blob/main/src/route_choice/recursive_logit.py) implements the standard recursive logit model. Internally, it can solve for the node values using fixed-point iteration (see [`FixedPointSolver`](https://github.com/ben-hudson/pytorch-choice-models/blob/main/src/route_choice/layers/fixed_point_solve.py)) or value iteration ([`ValueIterationSolver`](https://github.com/ben-hudson/pytorch-choice-models/blob/main/src/route_choice/layers/value_iteration_solve.py)).

### Examples
This test fits the model to a synthetic dataset generated on a small network with 11 nodes and 19 edges.
- [test_recursive_logit.py](https://github.com/ben-hudson/pytorch-choice-models/blob/main/tests/test_recursive_logit.py)

## Other libraries
The purpose of this library is to provide discrete choice models that can be fitted using PyTorch optimizers.

If you're looking for a discrete choice Python library in general, [PyLogit](https://github.com/timothyb0912/pylogit) implements many models with [examples](https://github.com/timothyb0912/pylogit/tree/master/examples/notebooks)!
