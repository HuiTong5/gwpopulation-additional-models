import numpy as np

from gwpopulation.models.redshift import _Redshift
from gwpopulation.utils import powerlaw, truncnorm

xp = np

class Transition_chi_eff:
    @property
    def variable_names(self):
        vars = ["m_t", "w", "log_sigma_chi_eff_low", "mu_chi_eff_low", "log_sigma_chi_eff_high", "mu_chi_eff_high", "xi_chi_eff"]
        return vars

    def __call__(self, dataset, *args, **kwargs):
        m_t = kwargs['m_t']
        w = kwargs['w']
        log_sigma_chi_eff_low = kwargs['log_sigma_chi_eff_low']
        mu_chi_eff_low = kwargs['mu_chi_eff_low']
        log_sigma_chi_eff_high = kwargs['log_sigma_chi_eff_high']
        mu_chi_eff_high = kwargs['mu_chi_eff_high']
        xi_chi_eff = kwargs['xi_chi_eff']

        p_chi = (dataset['mass_1']<m_t) * truncnorm(dataset['chi_eff'], mu_chi_eff_low, xp.exp(log_sigma_chi_eff_low), 1, -1)
        p_chi += (dataset['mass_1']>=m_t) * (xi_chi_eff* self.p_Uniform_chi_eff(dataset['chi_eff'], w)+ (1-xi_chi_eff)*truncnorm(dataset['chi_eff'], mu_chi_eff_high, xp.exp(log_sigma_chi_eff_high), 1, -1))

        return p_chi
    
    def p_Uniform_chi_eff(self, chi_eff, width):

        p = 1/(2*width)
        p += (chi_eff<-width) * (xp.exp(-(xp.abs(chi_eff)-width)**2/(2*0.1**2))-(1/(2*width)))
        p += (chi_eff>width) * (xp.exp(-(xp.abs(chi_eff)-width)**2/(2*0.1**2))-(1/(2*width)))
        return p


class MadauDickinsonRedshift(_Redshift):
    r"""
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270 (33)
    See https://arxiv.org/abs/2003.12152 (2) for the normalisation

    The parameterisation differs a little from there, we use

    .. math::
        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) &= \frac{(1 + z)^\gamma}{1 + (\frac{1 + z}{1 + z_p})^\kappa}

    Parameters
    ----------
    gamma: float
        Slope of the distribution at low redshift
    kappa: float
        Slope of the distribution at high redshift
    z_peak: float
        Redshift at which the distribution peaks.
    z_max: float, optional
        The maximum redshift allowed.
    """

    variable_names = ["gamma", "kappa", "z_peak"]

    def psi_of_z(self, redshift, **parameters):
        gamma = parameters["gamma"]
        kappa = parameters["kappa"]
        z_peak = parameters["z_peak"]
        psi_of_z = (1 + redshift) ** gamma / (
            1 + ((1 + redshift) / (1 + z_peak)) ** (gamma+kappa)
        )
        psi_of_z *= 1 + (1 + z_peak) ** (-gamma-kappa)
        return psi_of_z


def total_four_volume(lamb, analysis_time, max_redshift=2.3):
    from astropy.cosmology import Planck15

    redshifts = xp.linspace(0, max_redshift, 1000)
    psi_of_z = (1 + redshifts) ** lamb
    normalization = 4 * xp.pi / 1e9 * analysis_time
    total_volume = (
        xp.trapz(
            Planck15.differential_comoving_volume(redshifts).value
            / (1 + redshifts)
            * psi_of_z,
            redshifts,
        )
        * normalization
    )
    return total_volume