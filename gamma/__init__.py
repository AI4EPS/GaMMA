"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from ._gaussian_mixture import GaussianMixture
from ._bayesian_mixture import BayesianGaussianMixture
from ._gaussian_mixture import calc_time, calc_amp, calc_mag


__all__ = ['GaussianMixture',
           'BayesianGaussianMixture',
           'calc_time',
           'calc_amp',
           'calc_mag',
           'utils']
