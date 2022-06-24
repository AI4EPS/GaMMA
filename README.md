# *GaMMA*: *Ga*ussian *M*ixture *M*odel *A*ssociation 

[![](https://github.com/wayneweiqiang/GaMMA/workflows/documentation/badge.svg)](https://wayneweiqiang.github.io/GaMMA)
[![](https://github.com/wayneweiqiang/GaMMA/workflows/wheels/badge.svg)](https://wayneweiqiang.github.io/GaMMA)

## 1. Install
```bash
pip install git+https://github.com/wayneweiqiang/GaMMA.git
```

The implementation is based on the [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html#gmm) in [scikit-learn](https://scikit-learn.org/stable/index.html)

## 2. Related papers
- Zhu, Weiqiang et al. "Earthquake Phase Association using a Bayesian Gaussian Mixture Model." (2021)
- Zhu, Weiqiang, and Gregory C. Beroza. "PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method." arXiv preprint arXiv:1803.03211 (2018).
![Method](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/diagram_gamma_annotated.png)

## 3. Examples

- Hyperparameters:
  - **dbscan_eps** (default = 10.0s): The maximum time between two picks for one to be considered as in the neighborhood of the other. See details in [DBSCAN](https://https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
  - **dbscan_min_samples** (default = 3): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
  - **min_picks_per_eq** (default = 10): Minimum picks for associated earthquakes.
  - **oversampling_factor** (default = 10): The initial number of clusters is determined by (Number of picks)/(Number of stations) * (oversampling factor).
  - **use_amplitude** (default = True): If using amplitude information.
  - **z(km)** (default = [0, 40]): The range of earthquake depth during association. 
  - **xlim_degree**, **ylim_degree**: The longitude and latitude range of the research region.

Note: DBSCAN is used to cut picks into small windows to speedup association.


- Synthetic Example

See details in the [notebook](https://github.com/wayneweiqiang/GaMMA/blob/master/docs/example_synthetic.ipynb): [example_synthetic.ipynb](example_synthetic.ipynb)

![Association result](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/result_eq05_err0.0_fp0.0_amp1.png)

- Real Example using [PhaseNet](https://wayneweiqiang.github.io/PhaseNet/) picks

See details in the [notebook](https://github.com/wayneweiqiang/GaMMA/blob/master/docs/example_phasenet.ipynb): [example_phasenet.ipynb](example_phasenet.ipynb)

- Real Example using [Seisbench](https://github.com/seisbench/seisbench)

See details in the [notebook](https://github.com/seisbench/seisbench/blob/main/examples/03c_catalog_seisbench_gamma.ipynb): [example_seisbench.ipynb](example_seisbench.ipynb)

![Associaiton result](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/2019-07-04T18-02-01.074.png)

More examples can be found in the earthquake detection workflow -- [QuakeFlow](https://wayneweiqiang.github.io/QuakeFlow/)
