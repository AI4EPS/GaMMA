"""Gaussian Mixture Model."""

# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import numpy as np

from scipy import linalg, optimize

from ._base import BaseMixture, _check_shape
from ..utils import check_array
from ..utils.extmath import row_norms
from ..utils.validation import _deprecate_positional_args


###############################################################################
# Gaussian mixture shape checkers used by the GaussianMixture class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like of shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    _check_shape(weights, (n_components,), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights


def _check_means(means, n_components, n_samples, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_samples, n_features), 'means')
    return means


def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be "
                         "positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (np.allclose(precision, precision.T) and
            np.all(linalg.eigvalsh(precision) > 0.)):
        raise ValueError("'%s precision' should be symmetric, "
                         "positive-definite" % covariance_type)


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)


def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : string

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(precisions, dtype=[np.float64, np.float32],
                             ensure_2d=False,
                             allow_nd=covariance_type == 'full')

    precisions_shape = {'full': (n_components, n_features, n_features),
                        'tied': (n_features, n_features),
                        'diag': (n_components, n_features),
                        'spherical': (n_components,)}
    _check_shape(precisions, precisions_shape[covariance_type],
                 '%s precision' % covariance_type)

    _check_precisions = {'full': _check_precisions_full,
                         'tied': _check_precision_matrix,
                         'diag': _check_precision_positivity,
                         'spherical': _check_precision_positivity}
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions


###############################################################################
# Gaussian mixture parameters estimators (used by the M-Step)

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, _, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances


def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    covariance.flat[::len(covariance) + 1] += reg_covar
    return covariance


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

    # n_components, _, n_features = means.shape
    # covariances = np.empty((n_components, n_features))
    # for k in range(n_components):
    #     diff = X - means[k]
    #     covariances[k] = np.diag(np.dot(resp[:, k] * diff.T, diff)) / nk[k]
    #     covariances[k] += reg_covar
    # return covariances

def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical variance values.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    variances : array, shape (n_components,)
        The variance values of each components.
    """
    return _estimate_gaussian_covariances_diag(resp, X, nk,
                                               means, reg_covar).mean(1)


def calc_time(center, station_locs, phase_type, vel={"p":6.0, "s":6.0/1.75}):
    """
    center: (loc, t)
    """
    v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
    t = np.linalg.norm(center[:,:-1] - station_locs, axis=-1, keepdims=True) / v + center[:,-1:]
    return t

def calc_mag(data, center, station_locs, weight, min=-2, max=8):
    """
    center: (loc, t)
    data: (n_sample, amp)
    """
    dist = np.linalg.norm(center[:,:-1] - station_locs, axis=-1, keepdims=True)
    # mag_ = ( data - 2.48 + 2.76 * np.log10(dist) )
    ## Picozzi et al. (2018) A rapid response magnitude scale...
    c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
    mag_ = (data - c0 - c3*np.log10(np.maximum(dist, 0.1)))/c1 + 3.5
    ## Atkinson, G. M. (2015). Ground-Motion Prediction Equation...
    # c0, c1, c2, c3, c4 = (-4.151, 1.762, -0.09509, -1.669, -0.0006)
    # mag_ = (data - c0 - c3*np.log10(dist))/c1
    #mag = np.sum(mag_ * weight) / (np.sum(weight)+1e-6)
    mu = np.sum(mag_ * weight) / (np.sum(weight)+1e-6)
    std = np.sqrt(np.sum((mag_-mu)**2 * weight) / (np.sum(weight)+1e-12))
    idx = (np.abs(mag_ - mu) <= std)
    mag = np.sum(mag_[idx] * weight[idx]) / (np.sum(weight[idx])+1e-6)
    mag = np.clip(mag, min, max)
    return mag

def calc_amp(mag, center, station_locs):
    """
    center: (loc, t)
    """
    dist = np.linalg.norm(center[:,:-1] - station_locs, axis=-1, keepdims=True)
    # logA = mag + 2.48 - 2.76 * np.log10(dist)
    ## Picozzi et al. (2018) A rapid response magnitude scale...
    c0, c1, c2, c3 = 1.08, 0.93, -0.015, -1.68
    logA = c0 + c1*(mag-3.5) + c3*np.log10(np.maximum(dist, 0.1))
    ## Atkinson, G. M. (2015). Ground-Motion Prediction Equation...
    # c0, c1, c2, c3, c4 = (-4.151, 1.762, -0.09509, -1.669, -0.0006)
    # logA = c0 + c1*mag + c3*np.log10(dist)
    return logA


## L2 norm
def diff_and_grad(vars, data, station_locs, phase_type, vel={"p":6.0, "s":6.0/1.75}):
    """
    data: (n_sample, t)
    """
    v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
    # loc, t = vars[:,:-1], vars[:,-1:]
    dist = np.sqrt(np.sum((station_locs - vars[:,:-1])**2, axis=1, keepdims=True))
    y = dist/v - (data - vars[:,-1:])
    J = np.zeros([data.shape[0], vars.shape[1]])
    J[:, :-1] = (vars[:,:-1] - station_locs)/(dist + 1e-6)/v
    J[:, -1] = 1
    return y, J

def newton_method(vars, data, station_locs, phase_type, weight, max_iter=20, convergence=1):
    for i in range(max_iter): 
        prev = vars.copy()
        y, J = diff_and_grad(vars, data, station_locs, phase_type)
        JTJ = np.dot(J.T, weight * J)
        I = np.zeros_like(JTJ)
        np.fill_diagonal(I, 1e-3)
        vars -= np.dot(np.linalg.inv(JTJ + I) , np.dot(J.T, y * weight)).T
        if (np.sum(np.abs(vars - prev))) < convergence:
            return vars
    return vars

## l1 norm
# def loss_and_grad(vars, data, station_locs, phase_type, weights, vel={"p":6.0, "s":6.0/1.75}):
    
#     v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
#     vars = vars[np.newaxis, :]
#     dist = np.sqrt(np.sum((station_locs - vars[:,:-1])**2, axis=1, keepdims=True))
#     J = np.zeros([data.shape[0], vars.shape[1]])
#     J[:, :-1] = (vars[:,:-1] - station_locs)/(dist + 1e-6)/v
#     J[:, -1] = 1

#     loss = np.sum(np.abs(dist/v - (data[:,-1:] - vars[:,-1:])) * weights)
#     J = np.sum(np.sign(dist/v - (data[:,-1:] - vars[:,-1:])) * weights * J, axis=0, keepdims=True)

#     return loss, J

## Huber loss
def loss_and_grad(vars, data, station_locs, phase_type, weights, sigma=1, vel={"p":6.0, "s":6.0/1.75}):
    
    v = np.array([vel[p] for p in phase_type])[:, np.newaxis]
    vars = vars[np.newaxis, :]
    dist = np.sqrt(np.sum((station_locs - vars[:,:-1])**2, axis=1, keepdims=True))
    J = np.zeros([data.shape[0], vars.shape[1]])
    J[:, :-1] = (vars[:,:-1] - station_locs)/(dist + 1e-6)/v
    J[:, -1] = 1
    
    y = dist/v - (data[:,-1:] - vars[:,-1:])
    mask = np.squeeze(np.abs(y) > sigma)

    loss = np.sum( (sigma*np.abs(y[mask]) - 0.5*sigma**2) * weights[mask] ) \
           + np.sum( 0.5*y[~mask]**2 * weights[~mask] )
    J = np.sum( sigma*np.sign(y[mask]) * J[mask] * weights[mask], axis=0, keepdims=True ) \
        + np.sum( y[~mask] * J[~mask] * weights[~mask], axis=0, keepdims=True )  

    return loss, J


def l1_bfgs(vars, data, station_locs, phase_type, weight, max_iter=5, convergence=1e-3, bounds=None): 

    opt = optimize.minimize(loss_and_grad, np.squeeze(vars), method="L-BFGS-B", jac=True,
                            args=(data, station_locs, phase_type, weight),
                            options={"maxiter": max_iter, "gtol": convergence, "iprint": -1},
                            bounds=bounds)

    return opt.x[np.newaxis, :]


def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type,  station_locs,  phase_type, 
                                  loss_type="l2", centers_prev=None, bounds=None):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    centers_prev: (stations(x, y, ...), time, amp, ...)

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    # means = np.dot(resp.T, X) / nk[:, np.newaxis]
    # means = np.tile(means, [X.shape[0],1,1]).transpose((1,0,2))
    n_features = X.shape[1]

    if centers_prev is None:
        centers_prev = np.dot(resp.T, np.hstack([station_locs, X])) / nk[:, np.newaxis]
    centers = np.zeros_like(centers_prev) #x, y, t, amp, ...
    
    for i in range(len(centers_prev)):
        if n_features == 1:
            if loss_type == "l2":
                centers[i:i+1, :] = newton_method(centers_prev[i:i+1,:], X, station_locs, phase_type, resp[:,i:i+1])
            elif loss_type == "l1":
                centers[i:i+1, :] = l1_bfgs(centers_prev[i:i+1,:], X, station_locs, phase_type, resp[:,i:i+1], bounds=bounds)
            else:
                raise ValueError(f"loss_type = {loss_type} not in l1 or l2")
        elif n_features == 2:
            if loss_type == "l2":
                centers[i:i+1, :-1] = newton_method(centers_prev[i:i+1,:-1], X[:,0:1], station_locs, phase_type, resp[:,i:i+1])
            elif loss_type == "l1":
                centers[i:i+1, :-1] = l1_bfgs(centers_prev[i:i+1,:-1], X[:,0:1], station_locs, phase_type, resp[:,i:i+1], bounds=bounds)
            else:
                raise ValueError(f"loss_type = {loss_type} not in l1 or l2")
            centers[i:i+1, -1:] = calc_mag(X[:,1:2], centers[i:i+1,:-1], station_locs, resp[:,i:i+1])
        else:
            raise ValueError(f"n_features = {n_features} > 2!")
    
    means = np.zeros([resp.shape[1], X.shape[0], X.shape[1]])
    for i in range(len(centers)):
        if n_features == 1:
            means[i, :, :] = calc_time(centers[i:i+1, :], station_locs, phase_type)
        elif n_features == 2:
            means[i, :, 0:1] = calc_time(centers[i:i+1, :-1], station_locs, phase_type)
            means[i, :, 1:2] = calc_amp(centers[i:i+1, -1:], centers[i:i+1, :-1], station_locs)
        else:
            raise ValueError(f"n_features = {n_features} > 2!")

    covariances = {"full": _estimate_gaussian_covariances_full,
                   "tied": _estimate_gaussian_covariances_tied,
                   "diag": _estimate_gaussian_covariances_diag,
                   "spherical": _estimate_gaussian_covariances_spherical
                   }[covariance_type](resp, X, nk, means, reg_covar)

    return nk, means, covariances, centers


def _compute_precision_cholesky(covariances, covariance_type, max_covar=None):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type == 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    
    if max_covar is not None:
        non_zero = (np.abs(precisions_chol) != 0.0)
        precisions_chol[non_zero] = 1.0/(np.sqrt(max_covar) * np.tanh(1.0/precisions_chol[non_zero]/np.sqrt(max_covar)))
        precisions_chol[~non_zero] = 1.0/np.sqrt(max_covar)

    return precisions_chol


###############################################################################
# Gaussian mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)

    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        precisions = precisions_chol ** 2
        log_prob = (np.sum((means ** 2 * precisions), 1) -
                    2. * np.dot(X, (means * precisions).T) +
                    np.dot(X ** 2, precisions.T))
        # log_prob = np.empty((n_samples, n_components))
        # for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        #     y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        #     log_prob[:, k] = np.square(y)

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


class GaussianMixture(BaseMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:

        'full'
            each component has its own general covariance matrix
        'tied'
            all components share the same general covariance matrix
        'diag'
            each component has its own diagonal covariance matrix
        'spherical'
            each component has its own single variance

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, default='kmeans'
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weights_init : array-like of shape (n_components, ), default=None
        The user-provided initial weights.
        If it is None, weights are initialized using the `init_params` method.

    means_init : array-like of shape (n_components, n_features), default=None
        The user-provided initial means,
        If it is None, means are initialized using the `init_params` method.

    precisions_init : array-like, default=None
        The user-provided initial precisions (inverse of the covariance
        matrices).
        If it is None, precisions are initialized using the 'init_params'
        method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        In that case, 'n_init' is ignored and only a single initialization
        occurs upon the first call.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import GaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    >>> gm.means_
    array([[10.,  2.],
           [ 1.,  2.]])
    >>> gm.predict([[0, 0], [12, 3]])
    array([1, 0])

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.
    """
    @_deprecate_positional_args
    def __init__(self, n_components=1, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None, centers_init=None,
                 random_state=None, warm_start=False, 
                 station_locs=None, phase_type=None, phase_weight=None, 
                 dummy_comp=False, dummy_prob=0.01, loss_type="l1", bounds=None, max_covar=None,
                 verbose=0, verbose_interval=10):
        super().__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            dummy_comp=dummy_comp, dummy_prob=dummy_prob,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.centers_init = centers_init
        if station_locs is None:
            raise("Missing: station_locs")
        if phase_type is None:
            raise("Missing: phase_type")
        if phase_weight is None:
            phase_weight = np.ones([len(phase_type),1])
        self.station_locs = station_locs
        self.phase_type = np.squeeze(phase_type)
        self.phase_weight = np.squeeze(phase_weight)
        self.loss_type = loss_type
        self.bounds = bounds
        self.max_covar = max_covar

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        n_samples, n_features = X.shape
        if self.covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError("Invalid value for 'covariance_type': %s "
                             "'covariance_type' should be in "
                             "['spherical', 'tied', 'diag', 'full']"
                             % self.covariance_type)

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init,
                                               self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(self.precisions_init,
                                                     self.covariance_type,
                                                     self.n_components,
                                                     n_features)
        
        if n_features > 2:
            raise ValueError(f"n_features = {n_features} > 2! Only support 2 features (time, amplitude)")
        assert(self.covariance_type=='full')
        assert(self.station_locs.shape[0] == n_samples)
        assert(self.loss_type in ["l1", "l2"])
        _check_shape(self.phase_type, (n_samples, ), 'phase_type')
        _check_shape(self.phase_weight, (n_samples, ), 'phase_type')
        if self.init_params == "centers":
            assert(self.centers_init is not None)
        # if self.centers_init is not None:
        #     _check_shape(self.centers_init, (self.n_components, self.station_locs.shape[1] + n_features), 'centers_init')


    def _initialize_centers(self, X, random_state):

        n_samples, n_features = X.shape

        means = np.zeros([self.n_components, n_samples, n_features])
        for i in range(len(self.centers_init)):
            if n_features == 1: #(time,)
                means[i, :, :] = calc_time(self.centers_init[i:i+1, :], self.station_locs, self.phase_type)
            elif n_features == 2: #(time, amp)
                means[i, :, 0:1] = calc_time(self.centers_init[i:i+1, :], self.station_locs, self.phase_type)
                means[i, :, 1:2] = X[:,1:2] #calc_amp(3.0, self.centers_init[i:i+1, :], self.station_locs)
            else:
                raise ValueError(f"n_features = {n_features} > 2!")

        dist = np.linalg.norm(means - X, axis=-1)
        resp = np.exp(-dist).T
        resp /= resp.sum(axis=1)[:, np.newaxis]

        return resp

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances, centers = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type, 
            self.station_locs, self.phase_type, loss_type=self.loss_type, 
            centers_prev=None, bounds=self.bounds)
        weights /= n_samples

        # self.weights_ = (weights if self.weights_init is None else self.weights_init)
        # self.means_ = (means if self.means_init is None else self.means_init)
        # self.centers_ = (centers if self.centers_init is None else self.centers_init)
        self.weights_ = weights
        self.means_ = means
        self.centers_ = centers

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type, self.max_covar)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
        else:
            self.precisions_cholesky_ = self.precisions_init

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_, self.centers_ = (
            _estimate_gaussian_parameters(
                X, np.exp(log_resp), self.reg_covar, self.covariance_type, 
                self.station_locs, self.phase_type, loss_type=self.loss_type, 
                centers_prev=self.centers_, bounds=self.bounds))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type, self.max_covar)

    def _estimate_log_prob(self, X):
        prob =  _estimate_log_gaussian_prob(X, self.means_, self.precisions_cholesky_, self.covariance_type)
        if self.dummy_comp:
            prob[:,-1] = np.log(self.dummy_prob)
        return prob + np.log(self.phase_weight)[:,np.newaxis]

    def _estimate_log_weights(self):
        if self.dummy_comp:
            score = 0.1 #1.0/len(self.weights_)
            if self.weights_[-1] >= score:
                self.weights_[:-1] /= np.sum(self.weights_[:-1]) / (1-score)
                self.weights_[-1] = score
        return np.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_) = params

        # Attributes computation
        _, _, n_features = self.means_.shape

        if self.covariance_type == 'full':
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

        elif self.covariance_type == 'tied':
            self.precisions_ = np.dot(self.precisions_cholesky_,
                                      self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, _, n_features = self.means_.shape
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
