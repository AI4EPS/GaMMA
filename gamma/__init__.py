"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from ._gaussian_mixture import GaussianMixture
from ._bayesian_mixture import BayesianGaussianMixture
# from .seismic_ops import *


__all__ = ['GaussianMixture',
           'BayesianGaussianMixture',
           'seismic_ops',
           'utils']
