====
GGMM
====

Gaussian Mixture Model with support for heterogeneous missing and censored (upper limit) data.

Pure python.

About
-----

Gaussian mixture models (GMMs) consist of 
weighted sums of Gaussian probability distributions.
They are a flexible tool to describe observations, and can be used
for classification and model density approximation in the context of 
simulation-based inference.

Missing data can occur when no measurement of a given feature was taken.
In that case, the probability of a GMM density can be obtained 
by marginalisation.
This is implemented in ggmm analytically.
This is different to `pygmmis <https://github.com/pmelchior/pygmmis>`,
which approximates this situation with large measurement uncertainties.
This is different to `<https://github.com/avati/gmm-mcar>`,
which assumes that missing measurements occur uniformly randomly.

Upper limits can occur when the measurement of a given feature was not
sensitive enough.
In that case, the probability of a GMM density can be obtained by
marginalisation up to the upper limit.
This is implemented in ggmm analytically, and each data point can have
its own individual upper limit (heterogeneous).
This is different to typical censored GMMs, which assume a common 
upper limit for all data (homogeneous) (`see here for example <https://github.com/tranbahien/Truncated-Censored-EM>`).

For these cases, GGMM implements evaluating the PDF and log-PDF of a mixture.
GGMM does not implement finding the mixture parameters.

Why
---

GGMM can be used for likelihood-based inference (LBI) with
simulation-based inference (SBI) generating samples, a EM algorithm
identifying the GMM parameters, but applied to data with missing data or upper limits.

This is a common case for photometric flux measurements in astronomy.

.. image:: https://img.shields.io/pypi/v/ggmm.svg
        :target: https://pypi.python.org/pypi/ggmm

.. image:: https://github.com/JohannesBuchner/ggmm/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/JohannesBuchner/ggmm/actions/workflows/tests.yml

.. image:: https://img.shields.io/badge/docs-published-ok.svg
        :target: https://johannesbuchner.github.io/ggmm/
        :alt: Documentation Status

Usage
^^^^^

Read the full documentation at:

https://johannesbuchner.github.io/ggmm/


Licence
^^^^^^^

GPLv3 (see LICENCE file). If you require another license, please contact me.

