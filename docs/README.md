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

- Inputs (the same names as [ADLoc](https://github.com/AI4EPS/ADLoc) since 1.8):
  - **picks**: columns station_id, phase_time, phase_type ("P" or "S"), phase_score, phase_amplitude (m/s).
  - **stations**: columns station_id and the coordinates named in **dims** (default ["x_km", "y_km", "z_km"], z_km positive down).
  - With **dims** = ["longitude", "latitude", "z_km"], stations are given in degrees and one run can cover any area, up to the whole globe: picks link on the globe, each segment is projected around its own stations (azimuthal equidistant, as ADLoc does per event), its events are located within 300 km of those stations, and come back in longitude, latitude. Needs segment "link", which keeps each segment local.
  - Events come back with the coordinates named in dims, plus depth_km. Names from before 1.8 (id, timestamp, type, prob, amp, x(km), min_picks_per_eq, ...) still work and are translated with one warning.
- Hyperparameters:
  - **xlim_km**, **ylim_km**, **zlim_km**: the region (km); zlim_km also sets the depth where each event starts.
  - **use_amplitude** (default = True): If using amplitude information.
  - **vel** (default = {"P": 6.0, "S": 6.0 / 1.73}): velocity for P and S phases (uniform velocity model).
  - **eikonal** (optional): When specified, the 1D velocity model overrides the uniform velocity model:
  ```python
  {"vel": {"Z": [depths],
  "P": [p_velocities],
  "S": [s_velocities]},
  "h": grid_spacing,
  "xlim_km": x_range,
  "ylim_km": y_range,
  "zlim_km": z_range}
  ```
  - **segment** (default = "link"): how to cut a long sequence of picks into segments, which speeds up association a lot. "link": groups of picks that one event could produce (stations within 100 km whose time differences fit the distance between them), each at most **segment_max_picks** (default = 200) picks, split further in time. "dbscan": the hierarchical DBSCAN cut below, the default before 1.7.0. None: all picks in one segment. It replaces `use_dbscan`, which still works with a warning (True → "link", False → None; use "dbscan" for the cut before 1.7.0).
  - **dbscan_eps** (default = 10.0s): The maximum time between two picks for one to be considered as a neighbor of the other. See details in [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
  - **dbscan_min_samples** (default = 3): The number of samples in a neighborhood for a point to be considered as a core point. See details in [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
  - **oversampling_factor** (default = 10): The initial number of clusters is determined by (Number of picks)/(Number of stations)/(Inital points) * (oversampling factor).
  - **initial_points** (default=[1,1,1] for (x, y, z) directions): Initial earthquake locations (cluster centers). For a large area over 10 degrees, more initial points are helpful, such as [2,2,1].
  - **covariance_prior** (default = 5 × (1.5², 0.5²) = (11.25, 1.25)) and **degrees_of_freedom_prior** (default = 5): the Wishart prior on each event's time (s²) and amplitude (log10²) residuals, as in scikit-learn: degrees_of_freedom_prior pseudo-picks with spreads of 1.5 s and 0.5. A larger time spread merges events; a smaller one splits them.
  - Filtering low quality association: each event is relocated on its own picks; picks with residuals above the limits below are dropped and the event relocated again, until its picks stop changing. The event is kept if enough picks remain.
    - **min_picks**: Minimum picks for associated earthquakes. We can also specify minimum P or S picks:
      - **min_p_picks**: Minimum P-picks for associated earthquakes.
      - **min_s_picks**: Minimum S-picks for associated earthquakes.
    - **max_residual_time** (default = 1.0): Max phase time residual (s)
    - **max_residual_amplitude** (default = 1.0): Max phase amplitude residual (log10 scale)
    - The old names `max_sigma11` and `max_sigma22` still work with a warning, as max_residual_time and max_residual_amplitude; `max_sigma12` is ignored.

Note the association speed is controlled by **dbscan_eps** and **oversampling_factor**. Larger values are preferred, but at the expense of a slower association speed.

- Synthetic Example

![Association result](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/result_eq05_err0.0_fp0.0_amp1.png)

- Example using [PhaseNet](https://wayneweiqiang.github.io/PhaseNet/) picks

See details in the [notebook](https://github.com/wayneweiqiang/GaMMA/blob/master/docs/example_phasenet.ipynb): [example_phasenet.ipynb](example_phasenet.ipynb)

- Example using [Seisbench](https://github.com/seisbench/seisbench)

See details in the [notebook](https://github.com/seisbench/seisbench/blob/main/examples/03c_catalog_seisbench_gamma.ipynb): [example_seisbench.ipynb](example_seisbench.ipynb)

![Associaiton result](https://raw.githubusercontent.com/wayneweiqiang/GaMMA/master/docs/assets/2019-07-04T18-02-01.074.png)

More examples can be found in the earthquake detection workflow -- [QuakeFlow](https://ai4eps.github.io/QuakeFlow/)
