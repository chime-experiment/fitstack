"""Define models that can be fit to the source stack."""

import numpy as np

from . import priors
from . import utils


class Model(object):
    """Baseclass for emcee models.

    Attributes
    ----------
    param_name : list of str
        Names of the model parameters.
    param_spec : dict
        Dictionary that specifies the prior distribution for each parameter.
        The keys are the parameter names and the values are themselves
        dictionaries, i.e,
            {name: {
                "fixed": bool,
                "value": float,
                "prior": str,
                "kwargs": dict
            }, ...
        where "fixed" is if the parameter should be held fixed at the "value" entry,
        "prior" is the class name in the fitstack.priors module, and
        "kwargs" is a dictionary containing any keyword arguments accepted by
        the __init__ method of that class.
    param_name_fit : list of str
        Names of the model parameters that are allowed to vary.
    param_name_fixed : list of str
        Names of the model parameters that are fixed at their default value.
    priors : dict
        Dictionary where the keys are the parameter names and the values
        are classes that are used to evaluate the prior distributions.
        Note that only *fit parameters* have an entry in this dictionary.
    default_values : np.ndarray[nparam,]
        The default values for all parameters.
    fit_index : np.ndarray
        Index into the param axis that yields the parameters that are
        allowed to vary.
    """

    param_name = []
    _param_spec = {}

    def __init__(self, seed=None, **param_spec):
        """Initialize the model.

        Parameters
        ----------
        seed : int
            Seed to use for random number generation.
            If the seed is not provided, then a random
            seed will be taken from system entropy.
        param_spec : dict
            Specifies the prior distribution for each parameter.
            See the description of the class attribute of the
            same name for the required format.  If a parameter
            is not provided than the default values defined
            in the _param_spec class attribute will be used.
        """

        if seed is None:
            seed = np.random.SeedSequence().entropy

        self.seed = seed
        self.rng = np.random.Generator(np.random.SFC64(seed))

        self.param_spec = {}
        for name, default_spec in self._param_spec.items():
            self.param_spec[name] = param_spec.get(name, default_spec)

        self.priors = {}
        for name, spec in self.param_spec.items():
            if not spec["fixed"]:
                PriorDistribution = getattr(priors, spec["prior"])
                self.priors[name] = PriorDistribution(rng=self.rng, **spec["kwargs"])

        # Set several useful attributes
        self.default_values = np.array(
            [self.param_spec[name]["value"] for name in self.param_name]
        )

        self.fit_index = np.flatnonzero(
            [not self.param_spec[name]["fixed"] for name in self.param_name]
        )

        self.param_name_fit = [
            name for name in self.param_name if not self.param_spec[name]["fixed"]
        ]

        self.param_name_fixed = [
            name for name in self.param_name if self.param_spec[name]["fixed"]
        ]

    def set_data(self, **kwargs):
        """Save any ancillary data needed to evaluate the probabily distribution.

        Parameters
        ----------
        kwargs : {attr: value, ...}
            Each keyword argument will be saved as an attribute
            that can be accessed by the methods that evaluate the
            log of the probability of observing the data given the
            model parameters.
        """

        for key, val in kwargs.items():
            setattr(self, key, val)

    def draw_random_parameters(self):
        """Draw random parameter values from the prior distribution.

        Returns
        -------
        theta : list
            Random parameter values.
        """

        theta = [self.priors[name].draw_random() for name in self.param_name_fit]
        return theta

    def log_prior(self, theta):
        """Evaluate the log of the prior distribution.

        Parameters
        ----------
        theta : list
            Values for the fit parameters.

        Returns
        -------
        prob : float
            Logarithm of the prior distribution.  This will be 0 if the
            parameters are within the ranges specified in the boundary
            class attribute and -Inf if they are outside the ranges.
        """

        priors = [
            self.prior[name].evaluate(th)
            for name, th in zip(self.param_name_fit, theta)
        ]

        with np.errstate(divide="ignore", invalid="ignore"):
            log_prior = np.sum(np.log(priors))

        return log_prior

    def log_likelihood(self, theta):
        """Evaluate the log of the likelihood.

        Parameters
        ----------
        theta : list
            Values for the fit parameters.

        Returns
        -------
        logL : float
            Logarithm of the likelihood function.
        """

        theta_all = self.get_all_params(theta)

        mdl = self.model(theta_all)

        residual = np.ravel(self.data - mdl)

        return -0.5 * np.matmul(residual.T, np.matmul(self.inv_cov, residual))

    def log_probability(self, theta):
        """Evaluate log of the probability of observing the data given the parameters.

        Parameters
        ----------
        theta : list
            Values for the fit parameters.

        Returns
        -------
        prob : float
            The product of the likelihood function and the prior distribution
            for the parameter values provided.
        """

        lp = self.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + self.log_likelihood(theta)

    def get_all_param(self, theta):
        """Return both fixed and variable parameters in the correct order.

        Parameters
        ----------
        theta : list
            Values for the subset of parameters that are being fit.

        Returns
        -------
        theta_all : array
            Values for the full set of parameters in the order specified
            by the param_name attribute.  Parameters that are fixed are
            set to their default values.
        """

        theta_all = np.copy(self.default_values)
        theta_all[self.fit_index] = theta

        return theta_all

    @property
    def nparam(self):
        """The total number of parameters."""
        return len(self.param_name)

    @property
    def nfixed(self):
        """The number of parameters that are fixed at their default value."""
        return len(self.param_name_fixed)

    @property
    def nfit(self):
        """The number of parameters that are being fit."""
        return len(self.param_name_fit)


class ScaledShiftedTemplate(Model):
    """Scaled and shifted template model."""

    param_name = ["amp", "offset"]

    _param_spec = {
        "amp": {
            "fixed": False,
            "value": 1.0,
            "type": "uniform",
            "parameters": {
                "low": 0.0,
                "high": 10.0,
            },
        },
        "offset": {
            "fixed": False,
            "value": 0.0,
            "type": "uniform",
            "parameters": {
                "low": -1.0,
                "high": 1.0,
            },
        },
    }

    def model(self, theta, freq=None, transfer=None, template=None):
        """Evaluate the model consisting of a scaled and shifted template.

        .. math::

            S(\Delta \nu) = A H(\Delta \nu) \circledast T(\Delta \nu - \nu_{0})

        where `S` is the stacked signal as a function of
        frequency offset :math:`\Delta \nu`, `A` is the amplitude,
        :math:`\nu_{0}` is the central frequency offset,
        and `H` is the transfer function.

        Parameters
        ----------
        theta : [amp, offset]
            Two element list containing the amplitude and offset parameters.
            The template will be multiplied by the amplitude and then shifted by
            the offset in MHz.
        freq : np.ndarray[nfreq]
            Frequency offset in MHz.  If this was not provided, then will
            default to the `freq` attribute.
        transfer : np.ndarray[..., nfreq]
            Transfer function of the pipeline.  The template is convolved
            with the transfer function.  If this was not provided,
            then will default to the `transfer` attribute.
        template : np.ndarray[..., nfreq]
            Template for the signal.  If this was not provided, then
            will default to the `template` attribute.

        Returns
        -------
        model : np.ndarray[..., nfreq]
            Model for the signal, which is the template convolved with the
            transfer function, shifted by frequency offset, and scaled by
            the amplitude.
        """

        if freq is None:
            freq = self.freq

        if transfer is None:
            transfer = self.transfer

        if template is None:
            template = self.template

        amp, offset = theta

        model = utils.shift_and_convolve(
            freq, amp * template, offset=offset, kernel=transfer
        )

        return model


class DeltaFunction(ScaledShiftedTemplate):
    """Delta function model.

    Subclass of the scaled and shifted template model that uses a delta function
    as the underlying template.  Useful to compare to the scaled and shifted
    template model to determine if we are sensitive to signal beyond 0 MHz lag.
    """

    def set_data(self, **kwargs):
        """Save any ancillary data needed to evaluate the probabily distribution.

        Parameters
        ----------
        kwargs : {attr: value, ...}
            Each keyword argument will be saved as an attribute
            that can be accessed by the methods that evaluate the
            log of the probability of observing the data given the
            model parameters.
        """

        super().set_data(**kwargs)

        icenter = np.argmin(np.abs(self.freq))
        self.template = np.zeros_like(self.data)
        self.template[..., icenter] = 1.0


class Exponential(Model):
    """Exponential model."""

    param_name = ["amp", "offset", "scale"]

    _param_spec = {
        "amp": {
            "fixed": False,
            "value": 100.0,
            "type": "uniform",
            "parameters": {
                "low": 0.0,
                "high": 500.0,
            },
        },
        "offset": {
            "fixed": False,
            "value": 0.0,
            "type": "uniform",
            "parameters": {
                "low": -1.0,
                "high": 1.0,
            },
        },
        "scale": {
            "fixed": False,
            "value": 0.7,
            "type": "uniform",
            "parameters": {
                "low": 0.1,
                "high": 10.0,
            },
        },
    }

    def model(self, theta, freq=None, transfer=None):
        """Evaluate the exponential model.

        .. math::

            S(\Delta \nu) = A H(\Delta \nu) \circledast exp(-|\Delta \nu - \nu{0}| / s)

        where `S` is the stacked signal as a function of frequency offset
        :math:`\Delta \nu`, `A` is the amplitude, `s` is the scale,
        :math:`\nu_{0}` is the central frequency, and `H` is the transfer function.

        Parameters
        ----------
        theta : [amp, offset, scale]
            Three element list containing the model parameters, which are
            the amplitude, central frequency (in MHz), and scale (in MHz).
        freq : np.ndarray[nfreq]
            Frequency offset in MHz.  If this was not provided, then will
            default to the `freq` attribute.
        transfer : np.ndarray[..., nfreq]
            Transfer function of the pipeline.  The exponential is convolved
            with the transfer function.  If this was not provided,
            then will default to the `transfer` attribute.

        Returns
        -------
        model : np.ndarray[..., nfreq]
            Model for the signal.
        """
        if freq is None:
            freq = self.freq

        if transfer is None:
            transfer = self.transfer

        amp, offset, scale = theta

        # Evaluate the model
        model = utils.shift_and_convolve(
            freq, amp * np.exp(-np.abs(freq) / scale), offset=offset, kernel=transfer
        )

        return model
