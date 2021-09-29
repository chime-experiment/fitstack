"""Define models that can be fit to the source stack."""
import inspect

import numpy as np

from draco.util import tools

from . import priors
from . import signal
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

        defaults = self.default_param_spec()
        self.param_spec = {}
        for name in self.param_name:
            self.param_spec[name] = param_spec.get(name, defaults[name])

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
            self.priors[name].evaluate(th)
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

    def get_all_params(self, theta):
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

    @classmethod
    def default_param_spec(cls):
        """Get the default parameter specification.

        Combines the default parameter specifications in the _param_spec attribute
        of all classes in the MRO, with base-class values overridden when the
        parameter name is repeated in a subclass.  Hence, when creating a new class,
        the full parameter specification dictionary does not need to be repeated if
        only a few parameters are being modified.
        """

        param_spec = {}

        # Iterate over the reversed MRO and look for _param_spec attributes
        # which get added to a temporary dict. We go over the reversed MRO so
        # that values in base classes are overridden.
        for c in inspect.getmro(cls)[::-1]:

            if hasattr(c, "_param_spec"):

                for key, val in c._param_spec.items():
                    param_spec[key] = val

        return param_spec

    def forward_transform_sampler(self, sample: np.ndarray) -> np.ndarray:
        """Take a sample (or set of) and transform into the basis used by the sampler.

        Use this to transform into a basis that is more easily traversed by the sampler.
        Must be an inverse of `backward_transform_sampler`.

        Parameters
        ----------
        sample
            A 1D array containing a single sample, or a 2D array containing rows of
            samples.

        Returns
        -------
        transformed_samples
            The sample transformed into the samplers basis.
        """
        return sample

    def backward_transform_sampler(self, sample: np.ndarray) -> np.ndarray:
        """Take a sample (or set of) and transform from the basis used by the sampler.

        Use this to transform from a basis that is more easily traversed by the sampler.
        Must be an inverse of `forward_transform_sampler`.

        Parameters
        ----------
        sample
            A 1D array containing a single sample, or a 2D array containing rows of
            samples in the basis used by the sampler.

        Returns
        -------
        original_samples
            The sample(s) transformed into the original basis.
        """
        return sample

    def log_probability_sampler(self, theta: np.ndarray) -> float:
        """A log probability function in the sampler's basis.

        Parameters
        ----------
        theta
            Coordinate vector in the samplers basis.

        Returns
        -------
        lp
            The log probability of the sample.
        """
        return (
            self.log_probability(self.backward_transform_sampler(theta)) +
            self.log_transform_measure(theta)
        )

    def log_transform_measure(self, theta: np.ndarray) -> float:
        return 0.0


class ScaledShiftedTemplate(Model):
    """Scaled and shifted template model."""

    param_name = ["amp", "offset"]

    _param_spec = {
        "amp": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 10.0,
            },
        },
        "offset": {
            "fixed": False,
            "value": 0.0,
            "prior": "Uniform",
            "kwargs": {
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
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 500.0,
            },
        },
        "offset": {
            "fixed": False,
            "value": 0.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -1.0,
                "high": 1.0,
            },
        },
        "scale": {
            "fixed": False,
            "value": 0.7,
            "prior": "Uniform",
            "kwargs": {
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
        model_init = amp * np.exp(-np.abs(freq) / scale)

        model = utils.shift_and_convolve(
            freq, model_init, offset=offset, kernel=transfer
        )

        return model


class SimulationTemplate(Model):
    """Model consisting of a linear combination of templates from simulations."""

    param_name = ["offset", "omega", "b_HI", "b_g", "NL", "FoGh", "FoGg", "M_10"]

    _param_spec = {
        "offset": {
            "fixed": False,
            "value": 0.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -0.8,
                "high": 0.8,
            },
        },
        "omega": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 5.0,
            },
        },
        "b_HI": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 8.0,
            },
        },
        # This is the only parameter which is reasonably constrained.
        # Use scale = 0.03 for QSO, 0.13 for LRG, and 0.10 for ELG.
        "b_g": {
            "fixed": False,
            "value": 1.0,
            "prior": "Gaussian",
            "kwargs": {
                "loc": 1.00,
                "scale": 0.03,
            },
        },
        "NL": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -1.0,
                "high": 7.0,
            },
        },
        "FoGh": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 4.0,
            },
        },
        "FoGg": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 4.0,
            },
        },
        "M_10": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 25.0,
            },
        },
    }

    def __init__(
        self,
        pattern,
        pol=None,
        weight=None,
        combine=True,
        sort=True,
        derivs=None,
        factor=1e3,
        aliases=None,
        *args,
        **kwargs
    ):

        if derivs is None:
            derivs = {"lin": (-1.0, 1.0)}

        if aliases is None:
            aliases = {"shotnoise": "M_10", "lin": "N"}

        self._signal_template = signal.SignalTemplate.load_from_stackfiles(
            pattern,
            pol=pol,
            weight=weight,
            combine=combine,
            sort=sort,
            derivs=derivs,
            factor=factor,
            aliases=aliases,
        )

        super().__init__(*args, **kwargs)

    def model(self, theta, freq=None, transfer=None, pol_sel=None):

        if freq is None:
            freq = self.freq

        if transfer is None:
            transfer = self.transfer

        if pol_sel is None:
            pol_sel = self.pol_sel

        param_dict = {k: v for k, v in zip(self.param_name, theta)}

        offset = param_dict.pop("offset")

        model_init = self._signal_template.signal(**param_dict)[pol_sel]

        model = utils.shift_and_convolve(
            freq, model_init, offset=offset, kernel=transfer
        )

        return model


class SimulationTemplateFoG(SimulationTemplate):
    """Model based on templates from simulations convolved with a FoG damping kernel."""

    param_name = ["offset", "omega", "b_HI", "b_g", "NL", "FoGh", "FoGg", "M_10"]

    _param_spec = {
        "omega": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -5.0,
                "high": 5.0,
            },
        },
        "b_HI": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -5.0,
                "high": 5.0,
            },
        },
        "FoGh": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 8.0,
            },
        },
        "FoGg": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 8.0,
            },
        },
    }

    def __init__(
        self,
        pattern,
        pol=None,
        weight=None,
        combine=True,
        sort=True,
        derivs=None,
        convolutions=None,
        delay_range=None,
        factor=1e6,
        aliases=None,
        *args,
        **kwargs
    ):

        if aliases is None:
            aliases = dict(shotnoise="M_10")

        self._signal_template = signal.SignalTemplateFoG.load_from_stackfiles(
            pattern,
            pol=pol,
            weight=weight,
            combine=combine,
            sort=sort,
            derivs=derivs,
            convolutions=convolutions,
            delay_range=delay_range,
            factor=factor,
            aliases=aliases,
        )

        super(SimulationTemplate, self).__init__(*args, **kwargs)


class SimulationTemplateFoGAltParam(SimulationTemplateFoG):
    """Model based on templates from simulations convolved with a FoG damping kernel.

    This uses an alternative parameterization of (Omega, Omega x b_HI, ...) compared to
    the (Omega, b_HI, ...) parameterization used in the SimulationTemplate and
    SimulationTemplateFoG models.
    """

    param_name = ["offset", "omega", "omega_b_HI", "b_g", "NL", "FoGh", "FoGg", "M_10"]

    _param_spec = {
        "omega": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -5.0,
                "high": 5.0,
            },
        },
        "omega_b_HI": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -5.0,
                "high": 5.0,
            },
        },
    }

    def model(self, theta, freq=None, transfer=None, pol_sel=None):

        if freq is None:
            freq = self.freq

        if transfer is None:
            transfer = self.transfer

        if pol_sel is None:
            pol_sel = self.pol_sel

        param_dict = {k: v for k, v in zip(self.param_name, theta)}

        offset = param_dict.pop("offset")

        omega_bHI = param_dict.pop("omega_b_HI")
        param_dict["b_HI"] = omega_bHI * tools.invert_no_zero(param_dict["omega"])

        model_init = self._signal_template.signal(**param_dict)[pol_sel]

        model = utils.shift_and_convolve(
            freq, model_init, offset=offset, kernel=transfer
        )

        return model


class SimulationTemplateFoGTransform(SimulationTemplateFoG):
    """An FoG damped template that samples in a decorrelated basis.

    This uses an alternative basis of (Omega, Omega x b_HI, ...) for the sampler, but
    the results are returned in the original basis.
    """

    def forward_transform_sampler(self, sample: np.ndarray) -> np.ndarray:
        """Transform to an Omega, Omega_b_HI basis."""

        newsample = sample.copy()
        newsample[..., 2] = sample[..., 1] * sample[..., 2]
        return newsample

    def backward_transform_sampler(self, sample: np.ndarray) -> np.ndarray:
        """Transform to an Omega, Omega_b_HI basis."""

        newsample = sample.copy()
        newsample[..., 2] = sample[..., 2] / sample[..., 1]
        return newsample

    def log_transform_measure(self, theta: np.ndarray) -> float:
        return -np.log(np.abs(theta[..., 1]))