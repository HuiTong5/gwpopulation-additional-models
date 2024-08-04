"""
Implemented mass models
"""

import inspect

import numpy as np
import scipy.special as scs

from gwpopulation.utils import powerlaw, truncnorm
from gwpopulation.models.mass import BaseSmoothedMassDistribution, power_law_primary_mass_ratio

xp = np

def GaussianPeak(mass, mmin, mpp, sigpp, gaussian_mass_maximum=100):
    p_norm = truncnorm(mass, mu=mpp, sigma=sigpp, high=gaussian_mass_maximum, low=mmin)
    return p_norm

class PowerLawSmoothedMassDistribution(BaseSmoothedMassDistribution):
    """
    Powerlaw + peak model for two-dimensional mass distribution with low
    mass smoothing.

    https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha: float
        Powerlaw exponent for more massive black hole.
    beta: float
        Power law exponent of the mass ratio distribution.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation of the Gaussian component.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The Gaussian component is bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = power_law_primary_mass_ratio

class GaussianSmoothedMassDistribution(BaseSmoothedMassDistribution):
    """
    Powerlaw + peak model for two-dimensional mass distribution with low
    mass smoothing.

    https://arxiv.org/abs/1801.02699 Eq. (11) (T&T18)

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha: float
        Powerlaw exponent for more massive black hole.
    beta: float
        Power law exponent of the mass ratio distribution.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    lam: float
        Fraction of black holes in the Gaussian component.
    mpp: float
        Mean of the Gaussian component.
    sigpp: float
        Standard deviation of the Gaussian component.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The Gaussian component is bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = GaussianPeak

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)