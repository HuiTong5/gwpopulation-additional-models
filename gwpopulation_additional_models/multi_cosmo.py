"""
Cosmological functionality in :code:`GWPopulation` is based on the :code:`wcosmo` package.
For more details see the `wcosmo documentation <https://wcosmo.readthedocs.io/en/latest/>`_.

We provide a mixin class :func:`gwpopulation.experimental.cosmo_models.CosmoMixin` that
can be used to add cosmological functionality to a population model.
"""

import numpy as xp
from wcosmo import z_at_value
from wcosmo.astropy import WCosmoMixin, available
from wcosmo.utils import disable_units as wcosmo_disable_units
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
from gwpopulation.models.redshift import MadauDickinsonRedshift
from .eff_spin import Smoothed_transition_chi_eff



class multi_CosmoMixin:
    """
    Mixin class that provides cosmological functionality to a subclass.

    Parameters
    ==========
    cosmo_model: str
        The cosmology model to use. Default is :code:`Planck15`.
        Should be of :code:`wcosmo.available.keys()`.
    """

    def __init__(self, cosmo_model, suffix=None):
        wcosmo_disable_units()
        self.cosmo_model = cosmo_model
        self.suffix=suffix
        if self.cosmo_model == "FlatwCDM":
            if suffix != None:
                self.cosmology_names = ["H0_{self.suffix}", "Om0_{self.suffix}", "w0_{self.suffix}"]
            else:
                self.cosmology_names = ["H0", "Om0_mass", "w0_mass"]
        elif self.cosmo_model == "FlatLambdaCDM":
            if suffix != None:
                self.cosmology_names = ["H0_{self.suffix}", "Om0_{self.suffix}"]
            else:
                self.cosmology_names = ["H0_mass", "Om0_mass"]
        else:
            self.cosmology_names = []
        self._cosmo = available[cosmo_model]

    def cosmology(self, parameters):
        """
        Return the cosmology model given the parameters.

        Parameters
        ==========
        parameters: dict
            The parameters for the cosmology model.

        Returns
        =======
        wcosmo.astropy.WCosmoMixin
            The cosmology model.
        """
        if isinstance(self._cosmo, WCosmoMixin):
            return self._cosmo
        else:
            if self.suffix:
                cosmology_variables={key[:-(self.suffix+1)]:parameters[key] for key in self.cosmology_names}
            else:
                cosmology_variables={key:parameters[key] for key in self.cosmology_names}
            return self._cosmo(**cosmology_variables)

    # def detector_frame_to_source_frame(self, data, **parameters):
    #     r"""
    #     Convert detector frame samples to sourece frame samples given cosmological
    #     parameters. Calculate the corresponding
    #     :math:`\frac{d \theta_{\rm detector}}{d \theta_{\rm source}}` Jacobian term.
    #     This includes factors of :math:`(1 + z)` for redshifted quantities.

    #     Parameters
    #     ==========
    #     data: dict
    #         Dictionary containing the samples in detector frame.
    #     parameters: dict
    #         The cosmological parameters for relevant cosmology model.

    #     Returns
    #     =======
    #     samples: dict
    #         Dictionary containing the samples in source frame.
    #     jacobian: array-like
    #         The Jacobian term.
    #     """

    #     samples = dict()
    #     if "luminosity_distance" in data.keys():
    #         cosmo = self.cosmology(self.parameters)
    #         samples["redshift"] = z_at_value(
    #             cosmo.luminosity_distance,
    #             data["luminosity_distance"],
    #         )
    #         jacobian = cosmo.dDLdz(samples["redshift"])
    #     elif "redshift" not in data:
    #         raise ValueError(
    #             f"Either luminosity distance or redshift provided in detector frame to source frame samples conversion"
    #         )
    #     else:
    #         jacobian = xp.ones(data["redshift"].shape)

    #     for key in list(data.keys()):
    #         if key.endswith("_detector"):
    #             samples[key[:-9]] = data[key] / (1 + samples["redshift"])
    #             jacobian *= 1 + samples["redshift"]
    #         elif key != "luminosity_distance":
    #             samples[key] = data[key]

    #     return samples, jacobian

class cosmo_SinglePeakSmoothedMassDistribution(SinglePeakSmoothedMassDistribution, multi_CosmoMixin):
    @property
    def variable_names(self):
        vars = getattr(
            self.primary_model,
            "variable_names",
            inspect.getfullargspec(self.primary_model).args[1:],
        )
        vars += ["beta", "delta_m"]
        vars += self.cosmology_names
        vars = set(vars).difference(self.kwargs.keys())
        return vars

    def __init__(self, cosmo_model, suffix, mmin=2, mmax=100, cache=False):
        SinglePeakSmoothedMassDistribution.__init__(self, mmin=mmin, mmax=mmax, cache=cache)
        multi_CosmoMixin.__init__(self, cosmo_model=cosmo_model, suffix=suffix)

    def __call__(self, dataset, *args, **kwargs):
        cosmo = self.cosmology(**kwargs)
        jacobian = xp.ones(data["mass_1_detector"].shape)
         #detector to source frame
        pesudo_redshift = z_at_value(
                cosmo.luminosity_distance,
                dataset["luminosity_distance"],
            )
        samples=dict()
        samples['mass_1'] = dataset['mass_1_detector']/(1+pesudo_redshift)
        jacobian *= (1+pesudo_redshift) # m1_dector/m1_source
        samples['mass_ratio'] = dataset['mass_ratio']

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
        prob = p_m1 * p_q / jacobian # prob in detector frame

        return prob #detector frame mass probability p(m1d,q|dL, H0_m)
        
class cosmo_MadauDickinsonRedshift(MadauDickinsonRedshift, multi_CosmoMixin):
    def __init__(self, z_max, cosmo_model, suffix=suffix):
        multi_CosmoMixin.__init__(self, cosmo_model=cosmo_model, suffix=suffix)
        self.z_max = z_max
        self.zs = xp.linspace(1e-6, z_max, 2500)

    def psi_of_z(self, redshift, **parameters):
        r"""
        Redshift model from Fishbach+
        (`arXiv:1805.10270 <https://arxiv.org/abs/1805.10270>`_ Eq. (33))
        See Callister+ (`arXiv:2003.12152 <https://arxiv.org/abs/2003.12152>`_
        Eq. (2)) for the normalisation.

        .. math::

            \psi(z|\gamma, \kappa, z_p) = \frac{(1 + z)^\gamma}{1 + (\frac{1 + z}{1 + z_p})^\kappa}

        Parameters
        ----------
        gamma: float
            Slope of the distribution at low redshift, :math:`\gamma`.
        kappa: float
            Slope of the distribution at high redshift, :math:`\kappa`.
        z_peak: float
            Redshift at which the distribution peaks, :math:`z_p`.
        """
        gamma = parameters["gamma"]
        kappa = parameters["kappa"]
        z_peak = parameters["z_peak"]
        psi_of_z = (1 + redshift) ** gamma / (
            1 + ((1 + redshift) / (1 + z_peak)) ** (kappa+gamma)
        )
        psi_of_z *= 1 + (1 + z_peak) ** (-kappa-gamma)
        return psi_of_z

    def __call__(self, dataset, **kwargs):
        cosmo = self.cosmology(**kwargs)
        samples=dict()
        samples['redshift']=z_at_value(
                cosmo.luminosity_distance,
                dataset["luminosity_distance"],
            )
        jacobian = cosmo.dDLdz(samples["redshift"]) # dL/dz
        prob=self.probability(dataset=samples, **kwargs) / jacobian # prob in detector frame

        return prob #probability of dL, p(dL|H0_z)
    

class cosmo_Smoothed_transition_chi_eff(Smoothed_transition_chi_eff, multi_CosmoMixin):
    @property
    def variable_names(self):
        vars = ["m_t", "w", "log_sigma_chi_eff_low", "mu_chi_eff_low", "log_sigma_chi_eff_high", "mu_chi_eff_high", "xi_chi_eff"]
        vars += self.cosmology_names
        return vars

    def __init__(self, cosmo_model, suffix=suffix):
        multi_CosmoMixin.__init__(self, cosmo_model=cosmo_model, suffix=suffix)

    def __call__(self, dataset, *args, **kwargs):
        m_t = kwargs['m_t']
        w = kwargs['w']
        log_sigma_chi_eff_low = kwargs['log_sigma_chi_eff_low']
        mu_chi_eff_low = kwargs['mu_chi_eff_low']
        log_sigma_chi_eff_high = kwargs['log_sigma_chi_eff_high']
        mu_chi_eff_high = kwargs['mu_chi_eff_high']
        xi_chi_eff = kwargs['xi_chi_eff']

        cosmo = self.cosmology(**kwargs)
        samples=dict()
        samples['redshift']=z_at_value(
                cosmo.luminosity_distance,
                dataset["luminosity_distance"],
            )
        samples['mass_1'] = dataset['mass_1_detector']/(1+samples['redshift'])
        samples['chi_eff'] = dataset['chi_eff']

        f_HM = self.smoothed_transition_factor(samples['mass_1'], m_t) # The fraction of mergers above m_t
        f_uniform = self.xi_smoothed_transition_factor(samples['mass_1'], m_t, xi_chi_eff)# The fraction of uniform spin megers within the mergers above m_t? Why?
        p_chi = f_HM*(f_uniform*smoothed_uniform(samples['chi_eff'],w)+(1-f_uniform)*truncnorm(samples['chi_eff'], mu_chi_eff_high, 10**(log_sigma_chi_eff_high), 1, -1))+(1-f_HM)*truncnorm(samples['chi_eff'], mu_chi_eff_low, 10**(log_sigma_chi_eff_low), 1, -1)
        
        return p_chi # detector frame chi_eff probability, p(chi_eff| m1d, dL, H0_s)
