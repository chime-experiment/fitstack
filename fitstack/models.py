"""Define models that can be fit to the source stack."""

import inspect
import logging

import numpy as np

from draco.util import tools

from . import priors
from . import signal
from . import utils


logger = logging.getLogger(__name__)


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

    def __init__(self, seed=None, force_real=True, **param_spec):
        """Initialize the model.

        Parameters
        ----------
        seed : int
            Seed to use for random number generation.
            If the seed is not provided, then a random
            seed will be taken from system entropy.
        force_real : bool
            Force input datasets to be real. Assumes that input
            datasets have been previously examined to verify
            that imaginary parts are small and/or unimportant.
            Default: True.
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

        self.force_real = force_real

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

    def _re(self, x):
        return np.real(x) if self.force_real else x

    def set_data(self, **kwargs):
        """Save any ancillary data needed to evaluate the probability distribution.

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

    def log_likelihood(self, theta, amp=None):
        """Evaluate the log of the likelihood.

        Parameters
        ----------
        theta : list
            Values for the fit parameters.
        amp : float, optional
            Scale model by extra amplitude. Default: None.

        Returns
        -------
        logL : float
            Logarithm of the likelihood function.
        """

        theta_all = self.get_all_params(theta)

        mdl = self.model(theta_all)
        if amp is not None:
            mdl *= amp

        residual = np.ravel(self.data - mdl)

        return -0.5 * np.matmul(residual.T, np.matmul(self.inv_cov, residual))

    def negative_log_likelihood(self, theta, amp=None):
        """Evaluate the negative log of the likelihood.

        Parameters
        ----------
        theta : list
            Values for the fit parameters.
        amp : float, optional
            Scale model by extra amplitude. Default: None.

        Returns
        -------
        nlogL : float
            Negative logarithm of the likelihood function.
        """
        return -self.log_likelihood(theta, amp=amp)

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
        return self.log_probability(
            self.backward_transform_sampler(theta)
        ) + self.log_transform_measure(theta)

    def log_transform_measure(self, theta: np.ndarray) -> float:
        return 0.0


class NullModel(Model):
    """Model that just returns zero.

    Intended for assessing a zero-signal null hypothesis.
    """

    param_name = []

    def model(self, theta):
        """Evaluate model.

        Parameters
        ----------
        theta : array_like
            Array of input parameters (unused).

        Returns
        -------
        model : float
            Null model value (zero).
        """
        return 0.0


class ScaledShiftedTemplate(Model):
    """Scaled and shifted stacking template model."""

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
        r"""Evaluate the model consisting of a scaled and shifted stacking template.

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
        r"""Evaluate the exponential model.

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

    _template_class = signal.SignalTemplate
    _template_kwargs = ("symmetrize", "reverse")

    def __init__(
        self,
        pattern,
        pol=None,
        weight=None,
        combine=True,
        sort=True,
        derivs=None,
        factor=1e6,
        aliases=None,
        *args,
        **kwargs,
    ):

        if derivs is None:
            derivs = {"lin": (-1.0, 1.0)}

        if aliases is None:
            aliases = {"shotnoise": "M_10", "lin": "NL"}

        self._signal_template = self._template_class.load_from_stackfiles(
            pattern,
            pol=pol,
            weight=weight,
            combine=combine,
            sort=sort,
            derivs=derivs,
            factor=factor,
            aliases=aliases,
            **{k: v for k, v in kwargs.items() if k in self._template_kwargs},
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

    _template_class = signal.SignalTemplateFoG
    _template_kwargs = SimulationTemplate._template_kwargs + (
        "convolutions",
        "delay_range",
    )


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

    This uses an alternative basis replacing various parameters to decorrelate the
    chains:

    - `b_HI -> omega_b_HI = omega * b_HI`
    - `FoGh -> FoG+ = log(FoGh * FoGg) / 2`
    - `FoGg -> FoG- = log(FoGh / FoGg) / 2

    However, the chains are returned (and priors applied) in the original basis.

    Parameters
    ----------
    pattern
        Glob pattern to find the signal template modes.
    data_reverse
        Reverse the frequency offset axis in the data before evaluating the likelihood.
        This is useful for testing issues in the signal generation.
    """

    def __init__(self, pattern: str, data_reverse: bool = False, *args, **kwargs):
        self._data_reverse = data_reverse
        logger.debug(f"Reversing the data before sampling: {self._data_reverse}")
        super().__init__(pattern, *args, **kwargs)

    def forward_transform_sampler(self, sample: np.ndarray) -> np.ndarray:
        """Transform to an Omega, Omega_b_HI, FoG+, FoG- basis."""

        newsample = sample.copy()
        newsample[..., 2] = sample[..., 1] * sample[..., 2]

        # Transform to FoG+ and FoG- parameters for sampling
        newsample[..., 5] = 0.5 * np.log(sample[..., 5] * sample[..., 6])
        newsample[..., 6] = 0.5 * np.log(sample[..., 5] / sample[..., 6])

        return newsample

    def backward_transform_sampler(self, sample: np.ndarray) -> np.ndarray:
        """Transform to an Omega, Omega_b_HI, FoG+, FoG- basis."""

        newsample = sample.copy()
        newsample[..., 2] = sample[..., 2] / sample[..., 1]

        # Transform back to FoGh and FoGg
        newsample[..., 5] = np.exp(sample[..., 5] + sample[..., 6])
        newsample[..., 6] = np.exp(sample[..., 5] - sample[..., 6])

        return newsample

    def log_transform_measure(self, theta: np.ndarray) -> float:
        """The measure for the coordinate transform."""

        # The measure for the transform for Omega_b_HI
        measure = -np.log(np.abs(theta[..., 1]))

        # The log-measure for the transform for FoG+/- transform: 2 * FoGh * FoGg
        measure = 2 * theta[..., 5] + np.log(2.0)

        return measure

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

        if self._data_reverse:
            # Reverse the frequency axis
            data = self.data[..., ::-1]
            npol, nfreq = data.shape
            inv_cov = self.inv_cov.reshape(npol, nfreq, npol, nfreq)
            inv_cov = inv_cov[:, ::-1, :, ::-1].reshape(npol * nfreq, npol * nfreq)
        else:
            data = self.data
            inv_cov = self.inv_cov

        residual = np.ravel(data - mdl)

        return -0.5 * np.matmul(residual.T, np.matmul(inv_cov, residual))


class AutoConstant(Model):
    """Power spectrum model that's a constant in k with a free amplitude."""

    param_name = ["amp"]

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
    }

    def model(self, theta, k1D=None, transfer=None, template=None, pol_sel=None):
        """Evaluate constant-power-spectrum model.

        Parameters
        ----------
        theta : [amp]
            One-element list containing the model amplitude.
        k1D : np.ndarray[npol, nk]
            K values for each pol.  If not provided, method will use
            the `k1D` attribute.
        transfer, template, pol_sel
            Unused arguments.

        Returns
        -------
        model : np.ndarray[..., nk]
            Model for the signal.
        """

        if k1D is None:
            k1D = self.k1D

        amp = theta[0]

        model = np.full(k1D.shape, amp)

        return model


class AutoScaledTemplate(Model):
    """Scaled power spectrum template model."""

    param_name = ["amp"]

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
    }

    def model(self, theta, k1D=None, template=None, transfer=None, pol_sel=None):
        """Evaluate the model consisting of a scaled power spectrum template.

        The power spectrum template is simply multiplied by a free amplitude.

        Parameters
        ----------
        theta : [amp]
            One-element list containing the template amplitude.
        k1D : np.ndarray[npol,nk]
            K values for each pol. (Not actually used in model evaluation.)
        template : np.ndarray[..., nk]
            Signal template.
        transfer, pol_sel
            Unused arguments.

        Returns
        -------
        model : np.ndarray[..., nk]
            Model for the signal.
        """

        if k1D is None:
            k1D = self.k1D

        if template is None:
            template = self.template

        amp = theta[0]

        model = amp * self._re(template)

        return model


class AutoSimulationTemplate1D(Model):
    """Linear combination of 1D power spectrum templates from simulations.

    Note that Finger-of-God damping is *not* varied in this class:
    the `FoGh` parameter have no effect, and the `FoGs` parameter just
    switches between the alphaFoG=1 shot noise template (if `FoGs != 0`)
    or the alphaFoG=0 template (if `FoGs == 0`). To force the no-FoG
    form of shot noise, keep the `FoGs` parameter fixed to zero.

    The derived class `AutoSimulationTemplate1DFoG` should be used to
    vary Finger-of-God damping.
    """

    param_name = ["omega", "b_HI", "NL", "FoGh", "SN", "FoGs"]

    _param_spec = {
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
        "SN": {
            "fixed": True,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 4.0,
            },
        },
        "FoGs": {
            "fixed": True,
            "value": 0.0,
            "prior": "Uniform",
            "kwargs": {
                "low": 0.0,
                "high": 4.0,
            },
        },
    }

    _template_class = signal.AutoSignalTemplate1D
    _template_kwargs = ()

    def __init__(
        self,
        pattern,
        clustering_filename_pattern="*.h5",
        shotnoise_filename_pattern="*.h5",
        pol=None,
        combine=True,
        sort=False,
        factor=1,
        nbins=7,
        logbins=True,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self._signal_template = self._template_class.load_from_ps1Dfiles(
            pattern,
            clustering_filename_pattern=clustering_filename_pattern,
            shotnoise_filename_pattern=shotnoise_filename_pattern,
            pol=pol,
            combine=combine,
            factor=factor,
            nbins=nbins,
            logbins=logbins,
            force_real=self.force_real,
            **{k: v for k, v in kwargs.items() if k in self._template_kwargs},
        )

    def model(self, theta, k1D=None, template=None, transfer=None, pol_sel=None):
        """Evaluate the model.

        Parameters
        ----------
        theta : np.ndarray[6]
            Parameter values, ordered as
            ["omega", "b_HI", "NL", "FoGh", "SN", "FoGs"].
        k1D, template, transfer
            Unused arguments.
        pol_sel : np.ndarray
            Indices of pols to evaluate for.

        Returns
        -------
        model : np.ndarray[..., nk]
            Model for the signal.
        """

        if pol_sel is None:
            pol_sel = self.pol_sel

        param_dict = {k: v for k, v in zip(self.param_name, theta)}

        model = self._signal_template.signal_1D(**param_dict)[pol_sel]

        return model


class AutoSimulationTemplate1DFoG(AutoSimulationTemplate1D):
    """Power spectrum model with varying multiplicative FoG damping.

    To vary FoG damping in the clustering signal but use the no-FoG
    form of the shot noise template, keep the `FoGs` paramter fixed
    to 0 but vary the `SN` parameter.
    """

    param_name = ["omega", "b_HI", "NL", "FoGh", "SN", "FoGs"]

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
    }

    _template_class = signal.AutoSignalTemplate1DFoG


class AutoSimulationTemplate1DFoGTransform(AutoSimulationTemplate1DFoG):
    """Power spectrum model with FoG and more efficient sampling.

    This uses an alternative basis that makes the following replacement:

    - `b_HI -> omega_b_HI = omega * b_HI`

    However, the chains are returned (and priors applied) in the original basis.
    """

    def forward_transform_sampler(self, sample: np.ndarray) -> np.ndarray:
        """Transform to an Omega, Omega*b basis."""

        newsample = sample.copy()
        newsample[..., 1] = sample[..., 0] * sample[..., 1]

        return newsample

    def backward_transform_sampler(self, sample: np.ndarray) -> np.ndarray:
        """Transform to an Omega, b basis."""

        newsample = sample.copy()
        newsample[..., 1] = sample[..., 1] / sample[..., 0]

        return newsample

    def log_transform_measure(self, theta: np.ndarray) -> float:
        """The measure for the coordinate transform."""

        # The measure for the transform for Omega*b
        measure = -np.log(np.abs(theta[..., 0]))

        return measure


class AutoSimulationTemplate1D_Omega2(AutoSimulationTemplate1D):
    """Version of AutoSimulationTemplate1D that samples in Omega_HI^2.

    This uses a uniform prior on omega^2. However, this class is mostly
    intended for chi^2 minimization, for which the prior doesn't matter.
    """

    param_name = ["omega^2", "b_HI", "NL", "FoGh", "SN", "FoGs"]

    _param_spec = {
        "omega^2": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -400.0,
                "high": 400.0,
            },
        }
    }

    _template_class = signal.AutoSignalTemplate1D
    _template_kwargs = ()

    def model(self, theta, k1D=None, template=None, transfer=None, pol_sel=None):
        """Evaluate the model.

        Parameters
        ----------
        theta : np.ndarray[6]
            Parameter values, ordered as
            ["omega^2", "b_HI", "NL", "FoGh", "SN", "FoGs"].
        k1D, template, transfer
            Unused arguments.
        pol_sel : np.ndarray
            Indices of pols to evaluate for.

        Returns
        -------
        model : np.ndarray[..., nk]
            Model for the signal.
        """

        if pol_sel is None:
            pol_sel = self.pol_sel

        param_dict = {k: v for k, v in zip(self.param_name, theta)}

        omega2 = param_dict.pop("omega^2")
        # Need to allow omega to be complex so that omega^2 can be negative
        # when model is evaluated
        param_dict["omega"] = (omega2 + 0.0j) ** 0.5

        model = self._signal_template.signal_1D(**param_dict)[pol_sel]

        return model


class AutoSimulationTemplate1DFoG_Omega2(AutoSimulationTemplate1D_Omega2):
    """Version of AutoSimulationTemplate1DFoG that samples in Omega_HI^2.

    This uses a uniform prior on omega^2. However, this class is mostly
    intended for chi^2 minimization, for which the prior doesn't matter.
    """

    param_name = ["omega^2", "b_HI", "NL", "FoGh", "SN", "FoGs"]

    _param_spec = {
        "omega^2": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -400.0,
                "high": 400.0,
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
    }

    _template_class = signal.AutoSignalTemplate1DFoG


class AutoSimulationTemplate2Dto1D(Model):
    """Linear combination of 2D power spectrum templates from simulations."""

    param_name = ["omega", "b_HI", "NL", "FoGh", "M_10"]

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

    _template_class = signal.AutoSignalTemplate2D
    _template_kwargs = ()

    def __init__(
        self,
        pattern,
        filename_pattern=None,
        pol=None,
        weight=None,
        signal_mask=None,
        combine=True,
        sort=False,
        derivs=None,
        factor=1,
        aliases=None,
        nbins=10,
        logbins=True,
        slow_1d_binning=False,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        if derivs is None:
            derivs = {"lin": (-1.0, 1.0)}

        if aliases is None:
            aliases = {"shotnoise": "M_10", "lin": "NL"}

        self.slow_1d_binning = slow_1d_binning

        self._signal_template = self._template_class.load_from_ps2Dfiles(
            pattern,
            filename_pattern=filename_pattern,
            pol=pol,
            weight=weight,
            signal_mask=signal_mask,
            combine=combine,
            derivs=derivs,
            factor=factor,
            aliases=aliases,
            nbins=nbins,
            logbins=logbins,
            force_real=self.force_real,
            **{k: v for k, v in kwargs.items() if k in self._template_kwargs},
        )

    def model(self, theta, k1D=None, template=None, transfer=None, pol_sel=None):
        """Evaluate the model.

        Parameters
        ----------
        theta : np.ndarray[5]
            Parameter values, ordered as ["omega", "b_HI", "NL", "FoGh", "M_10"].
        k1D : np.ndarray[npol,nk]
            K values for each pol. (Not actually used in model evaluation.)
        template, transfer
            Unused arguments.
        pol_sel : np.ndarray
            Indices of pols to evaluate for.

        Returns
        -------
        model : np.ndarray[..., nk]
            Model for the signal.
        """

        if pol_sel is None:
            pol_sel = self.pol_sel

        param_dict = {k: v for k, v in zip(self.param_name, theta)}

        if self.slow_1d_binning:
            model = self._signal_template.signal_1D_slow(**param_dict)[pol_sel]
        else:
            model = self._signal_template.signal_1D(**param_dict)[pol_sel]

        return model


class AutoSimulationTemplate2Dto1DFoG(AutoSimulationTemplate2Dto1D):
    """Power spectrum model with varying FoG damping via multiplicative kernel."""

    param_name = ["omega", "b_HI", "NL", "FoGh", "M_10"]

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
    }

    _template_class = signal.AutoSignalTemplate2DFoG
    _template_kwargs = AutoSimulationTemplate2Dto1D._template_kwargs + (
        "convolutions",
        "kpara_range",
    )


class AutoSimulationTemplate2Dto1D_Omega2(AutoSimulationTemplate2Dto1D):
    """Version of AutoSimulationTemplate2Dto1D that samples in Omega_HI^2.

    This uses a uniform prior on omega^2. However, this class is mostly
    intended for chi^2 minimization, for which the prior doesn't matter.
    """

    param_name = ["omega^2", "b_HI", "NL", "FoGh", "M_10"]

    _param_spec = {
        "omega^2": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -25.0,
                "high": 25.0,
            },
        }
    }

    _template_class = signal.AutoSignalTemplate2D
    _template_kwargs = ()

    def model(self, theta, k1D=None, template=None, transfer=None, pol_sel=None):
        """Evaluate the model.

        Parameters
        ----------
        theta : np.ndarray[5]
            Parameter values, ordered as ["omega^2", "b_HI", "NL", "FoGh", "M_10"].
        k1D : np.ndarray[npol,nk]
            K values for each pol. (Not actually used in model evaluation.)
        template, transfer
            Unused arguments.
        pol_sel : np.ndarray
            Indices of pols to evaluate for.

        Returns
        -------
        model : np.ndarray[..., nk]
            Model for the signal.
        """

        if pol_sel is None:
            pol_sel = self.pol_sel

        param_dict = {k: v for k, v in zip(self.param_name, theta)}

        omega2 = param_dict.pop("omega^2")
        # Need to allow omega to be complex so that omega^2 can be negative
        # when model is evaluated
        param_dict["omega"] = (omega2 + 0.0j) ** 0.5

        if self.slow_1d_binning:
            model = self._signal_template.signal_1D_slow(**param_dict)[pol_sel]
        else:
            model = self._signal_template.signal_1D(**param_dict)[pol_sel]

        return model


class AutoSimulationTemplate2Dto1DFoG_Omega2(AutoSimulationTemplate2Dto1D_Omega2):
    """Version of AutoSimulationTemplate2Dto1DFoG that samples in Omega_HI^2.

    This uses a uniform prior on omega^2. However, this class is mostly
    intended for chi^2 minimization, for which the prior doesn't matter.
    """

    param_name = ["omega^2", "b_HI", "NL", "FoGh", "M_10"]

    _param_spec = {
        "omega^2": {
            "fixed": False,
            "value": 1.0,
            "prior": "Uniform",
            "kwargs": {
                "low": -25.0,
                "high": 25.0,
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
    }

    _template_class = signal.AutoSignalTemplate2DFoG
    _template_kwargs = AutoSimulationTemplate2Dto1D._template_kwargs + (
        "convolutions",
        "kpara_range",
    )
