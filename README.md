# *GaMMA*: *Ga*ussian *M*ixture *M*odel *A*ssociation 

[![](https://github.com/wayneweiqiang/GMMA/workflows/documentation/badge.svg)](https://wayneweiqiang.github.io/GMMA)
[![](https://github.com/wayneweiqiang/GMMA/workflows/pypi/badge.svg)](https://wayneweiqiang.github.io/GMMA)
[![](https://github.com/wayneweiqiang/GMMA/workflows/wheels/badge.svg)](https://wayneweiqiang.github.io/GMMA)

## 1. Install
```bash
pip install -i https://pypi.anaconda.org/zhuwq0/simple gmma
```

The implementation is based on the [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html#gmm) in [scikit-learn](https://scikit-learn.org/stable/index.html)

## 2. Related papers
- Zhu, Weiqiang et al. "Earthquake Phase Association using a Bayesian Gaussian Mixture Model." (2021)
- Zhu, Weiqiang, and Gregory C. Beroza. "PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method." arXiv preprint arXiv:1803.03211 (2018).
![Method](https://raw.githubusercontent.com/wayneweiqiang/GMMA/master/docs/assets/diagram_gmma_annotated.png)

## 3. Examples

- Synthetic Example

See details in the [notebook](https://github.com/wayneweiqiang/GMMA/blob/master/docs/example_phasenet.ipynb): [example_synthetic.ipynb](example_phasenet.ipynb)

![Association result](https://raw.githubusercontent.com/wayneweiqiang/GMMA/master/docs/assets/result_eq05_err0.0_fp0.0_amp1.png)

- Real Example using [PhaseNet](https://wayneweiqiang.github.io/PhaseNet/) picks

See details in the [notebook](https://github.com/wayneweiqiang/GMMA/blob/master/docs/example_phasenet.ipynb): [example_phasenet.ipynb](example_phasenet.ipynb)

![Associaiton result](https://raw.githubusercontent.com/wayneweiqiang/GMMA/master/docs/assets/2019-07-04T18-02-01.074.png)

More examples can be found in the earthquake detection workflow -- [QuakeFlow](https://wayneweiqiang.github.io/QuakeFlow/)
