"""Multivariate Gaussians with support for upper limits and missing data."""

__version__ = '2.0.1'
from .gaussian import Gaussian, pdfcdf
from .mixture import GaussianMixture
