# LiNG Discovery Algorithm

If you use LiNGD, install R and the muRty package and set the path to Rscript.

## Installation

```sh
pip install git+https://github.com/cdt15/lingd.git
```

## Usage

```python
from lingd import LiNGD

# create instance and fit to the data.
model = LiNGD()
model.fit(X)

# estimated results
print(model.adjacency_matrices_)

print(model.costs_)

print(model.is_stables_)

# bounds of causal effects
print(model.bound_of_causal_effect(1))
```

## Example

[lingd/examples/lingd.ipynb](./examples/lingd.ipynb)

## References

* Gustavo Lacerda, Peter Spirtes, Joseph Ramsey, and Patrik O. Hoyer. **Discovering cyclic causal models by independent components analysis**. *In Proceedings of the Twenty-Fourth Conference on Uncertainty in Artificial Intelligence (UAI'08)*. AUAI Press, Arlington, Virginia, USA, 366– 374.
