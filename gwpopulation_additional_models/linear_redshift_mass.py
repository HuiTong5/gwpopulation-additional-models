import numpy as xp
import scipy.special as scs
import inspect
from wcosmo import z_at_value
from wcosmo.astropy import WCosmoMixin, available
from wcosmo.utils import disable_units as wcosmo_disable_units
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift
from .eff_spin import Smoothed_transition_chi_eff, smoothed_uniform
from gwpopulation.utils import powerlaw, truncnorm


def redshift_linear_two_component_single(
    dataset, xi_redshift, alpha, mmin, mmax, lam, mpp, sigpp, gaussian_mass_maximum=100
):
    r"""
    Power law model for one-dimensional mass distribution with a Gaussian component.

    .. math::
        p(m) &= (1 - \lambda_m) p_{\text{pow}} + \lambda_m p_{\text{norm}}

        p_{\text{pow}}(m) &\propto m^{-\alpha} : m_\min \leq m < m_\max

        p_{\text{norm}}(m) &\propto \exp\left(-\frac{(m - \mu_{m})^2}{2\sigma^2_m}\right)

    Parameters
    ----------
    mass: array-like
        Array of mass values (:math:`m`).
    alpha: float
        Negative power law exponent for the black hole distribution (:math:`\alpha`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum black hole mass (:math:`m_\max`).
    lam: float
        Fraction of black holes in the Gaussian component (:math:`\lambda_m`).
    mpp: float
        Mean of the Gaussian component (:math:`\mu_m`).
    sigpp: float
        Standard deviation of the Gaussian component (:math:`\sigma_m`).
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
    """
    mass=dataset["mass_1"]
    redshift=dataset["redshift"]
    p_pow = powerlaw(mass, alpha=-alpha, high=mmax, low=mmin)
    p_norm = truncnorm(mass, mu=mpp*(1+xi_redshift*redshift), sigma=sigpp, high=gaussian_mass_maximum, low=mmin)
    prob = (1 - lam) * p_pow + lam * p_norm
    return prob

class redshift_linear_BaseSmoothedMassDistribution:
    """
    Generic smoothed mass distribution base class.

    Implements the low-mass smoothing and power-law mass ratio
    distribution. Requires p_m1 to be implemented.

    Parameters
    ==========
    mmin: float
        The minimum mass considered for numerical normalization
    mmax: float
        The maximum mass considered for numerical normalization
    """

    primary_model = None

    @property
    def variable_names(self):
        vars = getattr(
            self.primary_model,
            "variable_names",
            inspect.getfullargspec(self.primary_model).args[1:],
        )
        vars += ["beta", "delta_m"]
        vars = set(vars).difference(self.kwargs.keys())
        return vars

    @property
    def kwargs(self):
        return dict()

    def __init__(self, mmin=2, mmax=100, normalization_shape=(1000, 500), cache=True):
        self.mmin = mmin
        self.mmax = mmax
        self.m1s = xp.linspace(mmin, mmax, normalization_shape[0])
        self.qs = xp.linspace(0.001, 1, normalization_shape[1])
        self.redshifts = xp.linspace(0.001, 5, 500)
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.cache = cache

    def __call__(self, dataset, *args, **kwargs):
        beta = kwargs.pop("beta")
        mmin = kwargs.get("mmin", self.mmin)
        mmax = kwargs.get("mmax", self.mmax)
        if "jax" not in xp.__name__:
            if mmin < self.mmin:
                raise ValueError(
                    "{self.__class__}: mmin ({mmin}) < self.mmin ({self.mmin})"
                )
            if mmax > self.mmax:
                raise ValueError(
                    "{self.__class__}: mmax ({mmax}) > self.mmax ({self.mmax})"
                )
        delta_m = kwargs.get("delta_m", 0)
        p_m1 = self.p_m1(dataset, **kwargs, **self.kwargs)
        p_q = self.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)
        prob = p_m1 * p_q
        return prob

    def p_m1(self, dataset, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        delta_m = kwargs.pop("delta_m", 0)
        p_m = self.__class__.primary_model(dataset, **kwargs)
        p_m *= self.smoothing(
            dataset["mass_1"], mmin=mmin, mmax=self.mmax, delta_m=delta_m
        )
        self._cache_m1_norms(dataset["redshift"])
        p_m /= self.norm_p_m1(delta_m=delta_m, **kwargs)
        return p_m 

    def _cache_m1_norms(self, redshift):
        """
        Cache the information necessary for linear interpolation of the mass
        ratio normalisation
        """
        from gwpopulation.models.interped import _setup_interpolant

        self._mass_interpolant = _setup_interpolant(
            self.redshifts, redshift, kind="linear", backend=xp
        )

    def norm_p_m1(self, delta_m, **kwargs):
        """Calculate the normalisation factor for the primary mass"""
        mmin = kwargs.get("mmin", self.mmin)
        if "jax" not in xp.__name__ and delta_m == 0:
            return 1
        
        m1s_grid, redshift_grid = xp.meshgrid(self.m1s, self.redshifts) 
        p_m = self.__class__.primary_model({'mass_1':m1s_grid, 'redshift':redshift_grid}, **kwargs)
        p_m *= self.smoothing(m1s_grid, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        norms = xp.nan_to_num(xp.trapz(p_m, m1s_grid)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )
        return self._mass_interpolant(norms)

    def p_q(self, dataset, beta, mmin, delta_m):
        p_q = powerlaw(dataset["mass_ratio"], beta, 1, mmin / dataset["mass_1"])
        p_q *= self.smoothing(
            dataset["mass_1"] * dataset["mass_ratio"],
            mmin=mmin,
            mmax=dataset["mass_1"],
            delta_m=delta_m,
        )

        try:
            if self.cache:
                p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)
            else:
                self._cache_q_norms(dataset["mass_1"])
                p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)
        except (AttributeError, TypeError, ValueError):
            self._cache_q_norms(dataset["mass_1"])
            p_q /= self.norm_p_q(beta=beta, mmin=mmin, delta_m=delta_m)

        return xp.nan_to_num(p_q)

    def norm_p_q(self, beta, mmin, delta_m):
        """Calculate the mass ratio normalisation by linear interpolation"""
        p_q = powerlaw(self.qs_grid, beta, 1, mmin / self.m1s_grid)
        p_q *= self.smoothing(
            self.m1s_grid * self.qs_grid, mmin=mmin, mmax=self.m1s_grid, delta_m=delta_m
        )

        norms = xp.nan_to_num(xp.trapz(p_q, self.qs, axis=0)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )

        return self._q_interpolant(norms)

    def _cache_q_norms(self, masses):
        """
        Cache the information necessary for linear interpolation of the mass
        ratio normalisation
        """
        from gwpopulation.models.interped import _setup_interpolant

        self._q_interpolant = _setup_interpolant(
            self.m1s, masses, kind="linear", backend=xp
        )

    @staticmethod
    def smoothing(masses, mmin, mmax, delta_m):
        """
        Apply a one sided window between mmin and mmin + delta_m to the
        mass pdf.

        The upper cut off is a step function,
        the lower cutoff is a logistic rise over delta_m solar masses.

        See T&T18 Eqs 7-8
        Note that there is a sign error in that paper.

        S = (f(m - mmin, delta_m) + 1)^{-1}
        f(m') = delta_m / m' + delta_m / (m' - delta_m)

        See also, https://en.wikipedia.org/wiki/Window_function#Planck-taper_window
        """
        if "jax" in xp.__name__ or delta_m > 0.0:
            shifted_mass = xp.nan_to_num((masses - mmin) / delta_m, nan=0)
            shifted_mass = xp.clip(shifted_mass, 1e-6, 1 - 1e-6)
            exponent = 1 / shifted_mass - 1 / (1 - shifted_mass)
            window = scs.expit(-exponent)
            window *= (masses >= mmin) * (masses <= mmax)
            return window
        else:
            return xp.ones(masses.shape)


class redshift_linear_SinglePeakSmoothedMassDistribution(redshift_linear_BaseSmoothedMassDistribution):
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

    primary_model = redshift_linear_two_component_single

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)

