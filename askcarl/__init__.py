"""Multivariate Gaussians with support for upper limits and missing data."""

__version__ = '1.1.0'
from .gaussian import Gaussian, pdfcdf
from .mixture import GaussianMixture
