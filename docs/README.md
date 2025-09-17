# [*GaMMA*: *Ga*ussian *M*ixture *M*odel *A*ssociator](https://ai4eps.github.io/GaMMA)

[![](https://github.com/wayneweiqiang/GaMMA/workflows/documentation/badge.svg)](https://ai4eps.github.io/GaMMA)
[![](https://github.com/wayneweiqiang/GaMMA/workflows/wheels/badge.svg)](https://ai4eps.github.io/GaMMA)

## 1. Install
```bash
pip install git+https://github.com/wayneweiqiang/GaMMA.git
```

The implementation is based on the [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html#gmm) in [scikit-learn](https://scikit-learn.org/stable/index.html)

## 2. Related papers
- Zhu, W., McBrearty, I. W., Mousavi, S. M., Ellsworth, W. L., & Beroza, G. C. (2022). Earthquake phase association using a Bayesian Gaussian mixture model. Journal of Geophysical Research: Solid Earth, 127(5), e2021JB023249.

**Generating automatic picks**:
- Zhu, W., & Beroza, G. C. (2019). PhaseNet: a deep-neural-network-based seismic arrival-time picking method. Geophysical Journal International, 216(1), 261-273.

**Adding 1D velocity model support**:
- Xi, Z., Wei, S. S., Zhu, W., Beroza, G. C., Jie, Y., & Saloor, N. (2024). Deep learning for deep earthquakes: insights from OBS observations of the Tonga subduction zone. Geophysical Journal International, 238(2), 1073-1088.
- Jie, Y., Wei, S. S., Zhu, W., Freymueller, J. T., & Elliott, J. (2025). Background Seismicity and Aftershocks of the 2020-2021 Large Earthquakes at the Alaska Peninsula Revealed by a Deep-learning-based Catalog. Authorea Preprints.

![Method](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/diagram_gamma_annotated.png)

## 3. Examples

- Hyperparameters:
  - **use_amplitude** (default = True): If using amplitude information.
  - **vel** (default = {"p": 6.0, "s": 6.0 / 1.75}): velocity for P and S phases (uniform velocity model).
  - **eikonal** (optional): When specified, the 1D velocity model overrides the uniform velocity model:
  ```python
  {"vel": {"z": [depths], 
  "p": [p_velocities], 
  "s": [s_velocities]}, 
  "h": grid_spacing, 
  "xlim": x_range, 
  "ylim": y_range, 
  "zlim": z_range}
  ```
  - **use_dbscan**: If using dbscan to cut a long sequence of picks into segments. Using DBSCAN can significantly speed up associaiton using small windows.
  - **dbscan_eps** (default = 10.0s): The maximum time between two picks for one to be considered as a neighbor of the other. See details in [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
  - **dbscan_min_samples** (default = 3): The number of samples in a neighborhood for a point to be considered as a core point. See details in [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
  - **oversampling_factor** (default = 10): The initial number of clusters is determined by (Number of picks)/(Number of stations)/(Inital points) * (oversampling factor).
  - **initial_points** (default=[1,1,1] for (x, y, z) directions): Initial earthquake locations (cluster centers). For a large area over 10 degrees, more initial points are helpful, such as [2,2,1].
  - **covariance_prior** (default = (5, 5)): covariance prior of time and amplitude residuals. As the code supports only a uniform or 1D velocity model, a large covariance prior can be used to avoid splitting one event into multiple events.
  - Filtering low quality association
    - **min_picks_per_eq**: Minimum picks for associated earthquakes. We can also specify minimum P or S picks:
  	- **min_p_picks_per_eq**: Minimum P-picks for associated earthquakes.
  	- **min_s_picks_per_eq**: Minimum S-picks for associated earthquakes.
    - **max_sigma11**: Max phase time residual (s)
    - **max_sigma22**: Max phase amplitude residual (in *log* scale)
    - **max_sigma12**: Max covariance term. (Usually not used)

Note the association speed is controlled by **dbscan_eps** and **oversampling_factor**. Larger values are preferred, but at the expense of a slower association speed.

- Synthetic Example

![Association result](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/result_eq05_err0.0_fp0.0_amp1.png)

- Example using [PhaseNet](https://wayneweiqiang.github.io/PhaseNet/) picks

See details in the [notebook](https://github.com/wayneweiqiang/GaMMA/blob/master/docs/example_phasenet.ipynb): [example_phasenet.ipynb](example_phasenet.ipynb)

- Example using [Seisbench](https://github.com/seisbench/seisbench)

See details in the [notebook](https://github.com/seisbench/seisbench/blob/main/examples/03c_catalog_seisbench_gamma.ipynb): [example_seisbench.ipynb](example_seisbench.ipynb)

![Associaiton result](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/2019-07-04T18-02-01.074.png)

More examples can be found in the earthquake detection workflow -- [QuakeFlow](https://ai4eps.github.io/QuakeFlow/)
