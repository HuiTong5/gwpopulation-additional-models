"""
Implemented mass models
"""

import inspect

import numpy as np
import scipy.special as scs
from gwpopulation.utils import powerlaw, truncnorm
from gwpopulation.models.mass import double_power_law_primary_mass, BaseSmoothedMassDistribution

xp = np

class Chieff_BaseSmoothedMassDistribution:
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
        vars += ["beta", "delta_m", "m_t", "w", "log_sigma_chi_eff_low", "mu_chi_eff_low", "log_sigma_chi_eff_high", "mu_chi_eff_high", "xi_chi_eff"]
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
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.cache = cache

    def __call__(self, dataset, *args, **kwargs):
        beta = kwargs.pop("beta")
        mmin = kwargs.get("mmin", self.mmin)
        mmax = kwargs.get("mmax", self.mmax)
        m_t = kwargs.pop("m_t")
        w = kwargs.pop("w")
        log_sigma_chi_eff_low = kwargs.pop("log_sigma_chi_eff_low")
        mu_chi_eff_low = kwargs.pop("mu_chi_eff_low")
        log_sigma_chi_eff_high = kwargs.pop("log_sigma_chi_eff_high")
        mu_chi_eff_high = kwargs.pop("mu_chi_eff_high")
        xi_chi_eff = kwargs.pop("xi_chi_eff")

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
        p_chi_eff = self.p_chi_eff(dataset, m_t=m_t, w=w, log_sigma_chi_eff_low=log_sigma_chi_eff_low, mu_chi_eff_low=mu_chi_eff_low, log_sigma_chi_eff_high=log_sigma_chi_eff_high, mu_chi_eff_high=mu_chi_eff_high, xi_chi_eff=xi_chi_eff)

        prob = p_m1 * p_q * p_chi_eff
        return prob

    def p_Uniform_chi_eff(self, chi_eff, width):

        return 1/(2*width)* (chi_eff>=-width)* (chi_eff<=width)

    def p_chi_eff(self, dataset, m_t, w, log_sigma_chi_eff_low, mu_chi_eff_low,log_sigma_chi_eff_high, mu_chi_eff_high, xi_chi_eff):
        
        p_chi_eff = (dataset["mass_1"] <= m_t) * truncnorm(dataset["chi_eff"], mu_chi_eff_low, xp.exp(log_sigma_chi_eff_low), 1, -1)
        p_chi_eff += (dataset["mass_1"] > m_t) * (xi_chi_eff*self.p_Uniform_chi_eff(dataset["chi_eff"],w) + (1-xi_chi_eff)*truncnorm(dataset["chi_eff"], mu_chi_eff_high, xp.exp(log_sigma_chi_eff_high), 1, -1))

        return p_chi_eff

    def p_m1(self, dataset, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        delta_m = kwargs.pop("delta_m", 0)
        p_m = self.__class__.primary_model(dataset["mass_1"], **kwargs)
        p_m *= self.smoothing(
            dataset["mass_1"], mmin=mmin, mmax=self.mmax, delta_m=delta_m
        )
        norm = self.norm_p_m1(delta_m=delta_m, **kwargs)
        return p_m / norm

    def norm_p_m1(self, delta_m, **kwargs):
        """Calculate the normalisation factor for the primary mass"""
        mmin = kwargs.get("mmin", self.mmin)
        if "jax" not in xp.__name__ and delta_m == 0:
            return 1
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        norm = xp.nan_to_num(xp.trapz(p_m, self.m1s)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )
        return norm

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

def double_power_law_two_peak_primary_mass(
    mass,
    alpha_1,
    alpha_2,
    mmin,
    mmax,
    break_fraction,
    lam,
    lam_1,
    mpp_1,
    sigpp_1,
    mpp_2,
    sigpp_2,
    gaussian_mass_maximum=100,
):
    r"""
    Broken power-law with a Gaussian component.

    .. math::
        p(m | \alpha_1, \alpha_2, m_\min, m_\max, \delta, \lambda_m, \mu_m, \sigma_m) =
        (1 - \lambda_m) p_{\text{bpl}}(m | \alpha_1, \alpha_2, m_\min, m_\max, \delta)
        + \lambda_m p_{\text{norm}}(m | \mu_m, \sigma_m)

    .. math::
        p_{\text{bpl}}(m | \alpha_1, m_\min, m_\max, \delta) &\propto \begin{cases}
            m^{-\alpha_1} : m_\min \leq m < m_\min + \delta (m_\max - m_\min)\\
            m^{-\alpha_2} : m_\min + \delta (m_\max - m_\min) \leq m < m_\max
        \end{cases}

    .. math::
        p_{\text{norm}}(m | \mu_m, \sigma_m) \propto \exp\left(-\frac{(m - \mu_{m})^2}{2\sigma^2_m}\right)

    Parameters
    ----------
    mass: array-like
        Mass to evaluate probability at (:math:`m`).
    alpha_1: float
        Powerlaw exponent for more massive black hole below break (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for more massive black hole above break (:math:`\alpha_2`).
    break_fraction:float
        The fraction between mmin and mmax primary mass distribution breaks (:math:`\delta`).
    mmin: float
        Minimum black hole mass (:math:`m_\min`).
    mmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    lam: float
        Fraction of black holes in the Gaussian components.
    lam_1: float
        Fraction of black holes in the lower mass Gaussian component.
    mpp_1: float
        Mean of the lower mass Gaussian component.
    mpp_2: float
        Mean of the upper mass Gaussian component.
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component.
    sigpp_2: float
        Standard deviation of the upper mass Gaussian component.
    gaussian_mass_maximum: float, optional
        Upper truncation limit of the Gaussian component. (default: 100)
    """

    p_pow = double_power_law_primary_mass(
        mass=mass,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        mmin=mmin,
        mmax=mmax,
        break_fraction=break_fraction,
    )
    p_norm1 = truncnorm(
        mass, mu=mpp_1, sigma=sigpp_1, high=gaussian_mass_maximum, low=mmin
    )
    p_norm2 = truncnorm(
        mass, mu=mpp_2, sigma=sigpp_2, high=gaussian_mass_maximum, low=mmin
    )
    prob = (1 - lam) * p_pow + lam * lam_1 * p_norm1 + lam * (1 - lam_1) * p_norm2
    return prob

class MultiPeakBrokenPowerLawSmoothedMassDistribution(BaseSmoothedMassDistribution):
    """
    Broken power law for two-dimensional mass distribution with low
    mass smoothing.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha_1: float
        Powerlaw exponent for more massive black hole below break.
    alpha_2: float
        Powerlaw exponent for more massive black hole above break.
    beta: float
        Power law exponent of the mass ratio distribution.
    break_fraction: float
        Fraction between mmin and mmax primary mass distribution breaks at.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    lam: float
        Fraction of black holes in the Gaussian components.
    lam_1: float
        Fraction of black holes in the lower mass Gaussian component.
    mpp_1: float
        Mean of the lower mass Gaussian component.
    mpp_2: float
        Mean of the upper mass Gaussian component.
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component.
    sigpp_2: float
        Standard deviation of the upper mass Gaussian component.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The Gaussian component is bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = double_power_law_two_peak_primary_mass

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)


class Chieff_MultiPeakBrokenPowerLawSmoothedMassDistribution(Chieff_BaseSmoothedMassDistribution):
    """
    Broken power law for two-dimensional mass distribution with low
    mass smoothing.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha_1: float
        Powerlaw exponent for more massive black hole below break.
    alpha_2: float
        Powerlaw exponent for more massive black hole above break.
    beta: float
        Power law exponent of the mass ratio distribution.
    break_fraction: float
        Fraction between mmin and mmax primary mass distribution breaks at.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    lam: float
        Fraction of black holes in the Gaussian components.
    lam_1: float
        Fraction of black holes in the lower mass Gaussian component.
    mpp_1: float
        Mean of the lower mass Gaussian component.
    mpp_2: float
        Mean of the upper mass Gaussian component.
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component.
    sigpp_2: float
        Standard deviation of the upper mass Gaussian component.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The Gaussian component is bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = double_power_law_two_peak_primary_mass

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)



class Chieff_q_BaseSmoothedMassDistribution:
    """
    Generic smoothed mass distribution base class.

    Implements the low-mass smoothing and two power-law mass ratio
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
        vars += ["beta_low", "beta_high", "delta_m", "m_t", "w", "log_sigma_chi_eff_low", "mu_chi_eff_low", "log_sigma_chi_eff_high", "mu_chi_eff_high", "xi_chi_eff"]
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
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.cache = cache

    def __call__(self, dataset, *args, **kwargs):
        beta_low = kwargs.pop("beta_low")
        beta_high = kwargs.pop("beta_high")
        mmin = kwargs.get("mmin", self.mmin)
        mmax = kwargs.get("mmax", self.mmax)
        m_t = kwargs.pop("m_t")
        w = kwargs.pop("w")
        log_sigma_chi_eff_low = kwargs.pop("log_sigma_chi_eff_low")
        mu_chi_eff_low = kwargs.pop("mu_chi_eff_low")
        log_sigma_chi_eff_high = kwargs.pop("log_sigma_chi_eff_high")
        mu_chi_eff_high = kwargs.pop("mu_chi_eff_high")
        xi_chi_eff = kwargs.pop("xi_chi_eff")

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
        p_q = self.p_q(dataset, m_t=m_t, beta_low=beta_low, beta_high=beta_high, mmin=mmin, delta_m=delta_m)
        p_chi_eff = self.p_chi_eff(dataset, m_t=m_t, w=w, log_sigma_chi_eff_low=log_sigma_chi_eff_low, mu_chi_eff_low=mu_chi_eff_low, log_sigma_chi_eff_high=log_sigma_chi_eff_high, mu_chi_eff_high=mu_chi_eff_high, xi_chi_eff=xi_chi_eff)

        prob = p_m1 * p_q * p_chi_eff
        return prob

    def p_Uniform_chi_eff(self, chi_eff, width):

        return 1/(2*width)* (chi_eff>=-width)* (chi_eff<=width)

    def p_chi_eff(self, dataset, m_t, w, log_sigma_chi_eff_low, mu_chi_eff_low,log_sigma_chi_eff_high, mu_chi_eff_high, xi_chi_eff):
        
        p_chi_eff = (dataset["mass_1"] <= m_t) * truncnorm(dataset["chi_eff"], mu_chi_eff_low, xp.exp(log_sigma_chi_eff_low), 1, -1)
        p_chi_eff += (dataset["mass_1"] > m_t) * (xi_chi_eff*self.p_Uniform_chi_eff(dataset["chi_eff"],w) + (1-xi_chi_eff)*truncnorm(dataset["chi_eff"], mu_chi_eff_high, xp.exp(log_sigma_chi_eff_high), 1, -1))

        return p_chi_eff

    def p_m1(self, dataset, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        delta_m = kwargs.pop("delta_m", 0)
        p_m = self.__class__.primary_model(dataset["mass_1"], **kwargs)
        p_m *= self.smoothing(
            dataset["mass_1"], mmin=mmin, mmax=self.mmax, delta_m=delta_m
        )
        norm = self.norm_p_m1(delta_m=delta_m, **kwargs)
        return p_m / norm

    def norm_p_m1(self, delta_m, **kwargs):
        """Calculate the normalisation factor for the primary mass"""
        mmin = kwargs.get("mmin", self.mmin)
        if "jax" not in xp.__name__ and delta_m == 0:
            return 1
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        norm = xp.nan_to_num(xp.trapz(p_m, self.m1s)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )
        return norm

    def p_q(self, dataset, m_t, beta_low, beta_high, mmin, delta_m):
        p_q = (dataset["mass_1"] <= m_t) * powerlaw(dataset["mass_ratio"], beta_low, 1, mmin / dataset["mass_1"])
        p_q += (dataset["mass_1"] > m_t) * powerlaw(dataset["mass_ratio"], beta_high, 1, mmin / dataset["mass_1"])
        p_q *= self.smoothing(
            dataset["mass_1"] * dataset["mass_ratio"],
            mmin=mmin,
            mmax=dataset["mass_1"],
            delta_m=delta_m,
        )

        try:
            if self.cache:
                p_q /= self.norm_p_q(m_t=m_t, beta_low=beta_low, beta_high=beta_high, mmin=mmin, delta_m=delta_m)
            else:
                self._cache_q_norms(dataset["mass_1"])
                p_q /= self.norm_p_q(m_t=m_t, beta_low=beta_low, beta_high=beta_high, mmin=mmin, delta_m=delta_m)
        except (AttributeError, TypeError, ValueError):
            self._cache_q_norms(dataset["mass_1"])
            p_q /= self.norm_p_q(m_t=m_t, beta_low=beta_low, beta_high=beta_high, mmin=mmin, delta_m=delta_m)

        return xp.nan_to_num(p_q)

    def norm_p_q(self, m_t, beta_low, beta_high, mmin, delta_m):
        """Calculate the mass ratio normalisation by linear interpolation"""
        p_q = (self.m1s_grid <= m_t) * powerlaw(self.qs_grid, beta_low, 1, mmin / self.m1s_grid)
        p_q += (self.m1s_grid > m_t) * powerlaw(self.qs_grid, beta_high, 1, mmin / self.m1s_grid)
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

class Chieff_q_MultiPeakBrokenPowerLawSmoothedMassDistribution(Chieff_q_BaseSmoothedMassDistribution):
    """
    Broken power law for two-dimensional mass distribution with low
    mass smoothing.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha_1: float
        Powerlaw exponent for more massive black hole below break.
    alpha_2: float
        Powerlaw exponent for more massive black hole above break.
    beta_low: float
        Power law exponent of the mass ratio distribution for mass_1 lower than m_t.
    beta_high: float
        Power law exponent of the mass ratio distribution for mass_1 higher than m_t.
    break_fraction: float
        Fraction between mmin and mmax primary mass distribution breaks at.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    lam: float
        Fraction of black holes in the Gaussian components.
    lam_1: float
        Fraction of black holes in the lower mass Gaussian component.
    mpp_1: float
        Mean of the lower mass Gaussian component.
    mpp_2: float
        Mean of the upper mass Gaussian component.
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component.
    sigpp_2: float
        Standard deviation of the upper mass Gaussian component.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The Gaussian component is bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = double_power_law_two_peak_primary_mass

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)

class Broken_q_BaseSmoothedMassDistribution:
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
        vars += ["beta_low", "beta_high", "delta_m", "m_t"]
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
        self.dm = self.m1s[1] - self.m1s[0]
        self.dq = self.qs[1] - self.qs[0]
        self.m1s_grid, self.qs_grid = xp.meshgrid(self.m1s, self.qs)
        self.cache = cache

    def __call__(self, dataset, *args, **kwargs):
        beta_low = kwargs.pop("beta_low")
        beta_high = kwargs.pop("beta_high")
        m_t = kwargs.pop("m_t")
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
        p_q = self.p_q(dataset, beta_low=beta_low, beta_high=beta_high, m_t=m_t, mmin=mmin, delta_m=delta_m)
    
        prob = p_m1 * p_q
        return prob

    def p_m1(self, dataset, **kwargs):
        mmin = kwargs.get("mmin", self.mmin)
        delta_m = kwargs.pop("delta_m", 0)
        p_m = self.__class__.primary_model(dataset["mass_1"], **kwargs)
        p_m *= self.smoothing(
            dataset["mass_1"], mmin=mmin, mmax=self.mmax, delta_m=delta_m
        )
        norm = self.norm_p_m1(delta_m=delta_m, **kwargs)
        return p_m / norm

    def norm_p_m1(self, delta_m, **kwargs):
        """Calculate the normalisation factor for the primary mass"""
        mmin = kwargs.get("mmin", self.mmin)
        if "jax" not in xp.__name__ and delta_m == 0:
            return 1
        p_m = self.__class__.primary_model(self.m1s, **kwargs)
        p_m *= self.smoothing(self.m1s, mmin=mmin, mmax=self.mmax, delta_m=delta_m)

        norm = xp.nan_to_num(xp.trapz(p_m, self.m1s)) * (delta_m != 0) + 1 * (
            delta_m == 0
        )
        return norm

    def p_q(self, dataset, beta_low, beta_high, m_t, mmin, delta_m):
        p_q = xp.where(dataset["mass_1"]<=m_t, powerlaw(dataset["mass_ratio"], beta_low, 1, mmin / dataset["mass_1"]), powerlaw(dataset["mass_ratio"], beta_high, 1, mmin / dataset["mass_1"]))
        p_q *= self.smoothing(
            dataset["mass_1"] * dataset["mass_ratio"],
            mmin=mmin,
            mmax=dataset["mass_1"],
            delta_m=delta_m,
        )

        try:
            if self.cache:
                p_q /= self.norm_p_q(beta_low=beta_low, beta_high=beta_high, m_t=m_t, mmin=mmin, delta_m=delta_m)
            else:
                self._cache_q_norms(dataset["mass_1"])
                p_q /= self.norm_p_q(beta_low=beta_low, beta_high=beta_high, m_t=m_t, mmin=mmin, delta_m=delta_m)
        except (AttributeError, TypeError, ValueError):
            self._cache_q_norms(dataset["mass_1"])
            p_q /= self.norm_p_q(beta_low=beta_low, beta_high=beta_high, m_t=m_t, mmin=mmin, delta_m=delta_m)

        return xp.nan_to_num(p_q)

    def norm_p_q(self,  beta_low, beta_high, m_t, mmin, delta_m):
        """Calculate the mass ratio normalisation by linear interpolation"""
        p_q = xp.where(self.m1s_grid<=m_t, powerlaw(self.qs_grid, beta_low, 1, mmin / self.m1s_grid), powerlaw(self.qs_grid, beta_high, 1, mmin / self.m1s_grid))
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

class Broken_q_MultiPeakBrokenPowerLawSmoothedMassDistribution(Broken_q_BaseSmoothedMassDistribution):
    """
    Broken power law for two-dimensional mass distribution with low
    mass smoothing.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'mass_1' and 'mass_ratio'.
    alpha_1: float
        Powerlaw exponent for more massive black hole below break.
    alpha_2: float
        Powerlaw exponent for more massive black hole above break.
    beta: float
        Power law exponent of the mass ratio distribution.
    break_fraction: float
        Fraction between mmin and mmax primary mass distribution breaks at.
    mmin: float
        Minimum black hole mass.
    mmax: float
        Maximum mass in the powerlaw distributed component.
    lam: float
        Fraction of black holes in the Gaussian components.
    lam_1: float
        Fraction of black holes in the lower mass Gaussian component.
    mpp_1: float
        Mean of the lower mass Gaussian component.
    mpp_2: float
        Mean of the upper mass Gaussian component.
    sigpp_1: float
        Standard deviation of the lower mass Gaussian component.
    sigpp_2: float
        Standard deviation of the upper mass Gaussian component.
    delta_m: float
        Rise length of the low end of the mass distribution.

    Notes
    -----
    The Gaussian component is bounded between [`mmin`, `self.mmax`].
    This means that the `mmax` parameter is _not_ the global maximum.
    """

    primary_model = double_power_law_two_peak_primary_mass

    @property
    def kwargs(self):
        return dict(gaussian_mass_maximum=self.mmax)
