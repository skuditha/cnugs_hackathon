"""Scipy curve-fit helpers from the regression notebook pattern."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.optimize import curve_fit


def sine_model(x, a: float, b: float):
    """Simple sine model used in the bootcamp curve-fit exercise."""
    return a * np.sin(x) + b


def fit_curve(model_func: Callable, x, y, sigma=None, p0=None, absolute_sigma: bool = False):
    """Fit a model and return parameters, covariance, and parameter errors."""
    popt, pcov = curve_fit(model_func, x, y, sigma=sigma, p0=p0, absolute_sigma=absolute_sigma)
    perr = np.sqrt(np.diag(pcov))
    return popt, pcov, perr


def chi2_ndf(y, y_fit, n_parameters: int, sigma=None) -> float:
    """Compute chi^2/ndf. If sigma is omitted, unweighted residuals are used."""
    y = np.asarray(y)
    y_fit = np.asarray(y_fit)
    if sigma is None:
        chi_sq = np.sum((y - y_fit) ** 2)
    else:
        chi_sq = np.sum(((y - y_fit) / np.asarray(sigma)) ** 2)
    ndf = len(y) - n_parameters
    if ndf <= 0:
        raise ValueError("Number of degrees of freedom must be positive.")
    return float(chi_sq / ndf)
