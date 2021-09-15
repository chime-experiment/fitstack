import logging
import re
import glob
import os
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np
from scipy.fftpack import next_fast_len

from draco.util import tools
from draco.core.containers import FrequencyStackByPol, MockFrequencyStackByPol

from . import utils

logger = logging.getLogger(__name__)


class SignalTemplate:
    """Create signal templates from pre-simulated modes and input parameters.

    Parameters
    ----------
    derivs
        A dictionary of derivates expected, giving their name (key), and a tuple of the
        parameter difference used in the simulations (between the perturbed sim and the
        base values) and the fiducial value of the parameter.
    factor
        A scaling factor to apply to the sims. Unfortunately some of the sims were
        generated in mK rather than K, so the default value (`1e-3`) will scale the
        templates into Kelvin.
    aliases
        Allow the parameters to be given by more meaningful names.
    """

    def __init__(
        self,
        derivs: Optional[Dict[str, Tuple[float, float]]] = None,
        factor: float = 1e-3,
        aliases: Optional[Dict[str, str]] = None,
    ):

        if derivs is None:
            derivs = {
                "NL": (0.3, 1.0),
                "FoGh": (0.2, 1.0),
                "FoGg": (0.2, 1.0),
            }
        self._derivs = derivs
        self._factor = factor
        self._aliases = aliases if aliases is not None else {}

    @classmethod
    def load_from_stackfiles(
        cls,
        pattern: str,
        pol: List[str] = None,
        weight: np.ndarray = None,
        combine: bool = True,
        sort: bool = True,
        **kwargs,
    ):
        """Load the signal template from a set of stack files.

        This will load the stack files from each location and try and compile them into
        a set which can be used to generate signal templates.

        Parameters
        ----------
        pattern
            A glob pattern that isolates the base signal templates.
        pol
            The desired polarisations.
        weight
            The weight to use when averaging over polarisations.
            Must have shape [npol, nfreq].  Only relevant if combine is True.
        combine
            Add an element to the polarisation axis called I that
            is the weighted sum of the XX and YY polarisation.
        sort
            Sort the frequency offset axis in ascending order.
        **kwargs
            Arguments passed on to the constructor.
        """

        dirs = glob.glob(pattern)

        matching = {}

        # Find directories which match the right format
        for d in sorted(dirs):
            mo = re.search(r"_compderiv-([^\/]+)", d)

            if mo is None:
                print(f"Directory {d} does not match expected format, rejecting")
                continue

            key = mo.group(1)

            if key in matching:
                raise ValueError(
                    "Did not find a unique set of modes at this location. "
                    "You might need to refine the pattern."
                )

            d = Path(d)

            if not d.is_dir():
                raise ValueError("Glob must point to directories")

            matching[key] = Path(d)

        # For each directory load all the stacking files and combine them
        stacks = {}
        for key, d in matching.items():
            stack_files = sorted(list(d.glob("*.h5")))

            if len(stack_files) == 0:
                print("No files found at matching path.")
                continue

            mocks = utils.load_mocks(stack_files, pol=pol)
            mocks.weight[:] = weight[np.newaxis, :] if weight is not None else 1.0
            stacks[key] = utils.average_stacks(
                mocks, pol=mocks.pol, combine=combine, sort=sort
            )

        # Create the object and try and construct all the required templates from the
        # stacks
        self = cls(**kwargs)
        self._interpret_stacks(stacks)

        return self

    def _interpret_stacks(self, stacks: Dict[str, FrequencyStackByPol]):
        # Generate the required templates from the stacks

        # Find all entries that have the linear component structure
        compterms = [k.split("-")[1] for k in stacks.keys() if k.startswith("00")]

        stack_modes = {}

        # Get the first frequency axis as a reference
        self._freq = next(iter(stacks.values())).freq[:].copy()
        self._freq.flags.writeable = False

        def _check_load_stack(key):
            # Validate the stack and extract the template and its variance

            if key not in stacks:
                raise RuntimeError(f"Stack {key} was not loaded.")

            stack = stacks[key]

            if not np.array_equal(stack.freq[:], self._freq):
                raise RuntimeError(
                    f"Frequencies in stack {key} do not match reference."
                )

            return (
                self._factor * stack.stack[:],
                self._factor ** 2
                * tools.invert_no_zero(stack.attrs["num"] * stack.weight[:]),
            )

        # For all linear component terms load them and construct the various HI,g,v
        # combination terms
        for term in compterms:

            s00, v00 = _check_load_stack(f"00-{term}")
            s01, v01 = _check_load_stack(f"01-{term}")
            s10, v10 = _check_load_stack(f"10-{term}")
            s11, v11 = _check_load_stack(f"11-{term}")

            template_mean = np.zeros((4,) + s00.shape)
            template_var = np.zeros((4,) + s00.shape)

            # Calculate the template for each component
            template_mean[0] = s11 - s10 - s01 + s00  # Phg
            template_mean[1] = s10 - s00  # Phv
            template_mean[2] = s01 - s00  # Pvg
            template_mean[3] = s00  # Pvv

            # Calculate the variance of each component
            template_var[0] = v11 + v10 + v01 + v00
            template_var[1] = v10 + v00
            template_var[2] = v01 + v00
            template_var[3] = v00

            stack_modes[term] = (template_mean, template_var)

        self._stack_comp = {}
        self._stack_noncomp = {}
        self._stack_comp["base"] = stack_modes["base"]

        # For the expected derivative modes combine the perturbed entry and the base
        # templates to get the derivative templates
        for name, (delta, _) in self._derivs.items():

            if name not in stack_modes:
                raise RuntimeError(f"Expected derivative {name} but could not load it.")

            s, v = stack_modes[name]
            sb, vb = stack_modes["base"]

            # Calculate the finite difference derivative
            fd_mode = (s - sb) / delta
            fd_var = (v + vb) / delta ** 2

            self._stack_comp[name] = (fd_mode, fd_var)

        # Load any non-component type terms. These are terms which sit outside the usual
        # bias and Kaiser factors (such as shot noise)
        noncompterms = [k for k in stacks.keys() if "-" not in k]
        for term in noncompterms:
            self._stack_noncomp[term] = _check_load_stack(term)

    def signal(
        self, *, omega: float, b_HI: float, b_g: float, **kwargs: float
    ) -> np.ndarray:
        """Return the signal template for the given parameters.

        Parameters
        ----------
        omega
            Overall scaling.
        b_HI
            Scaling for the HI bias term.
        b_g
            Scaling for tracer bias term.
        **kwargs
            Values for all other derivative terms (e.g. NL) and non-component terms
            (e.g. shotnoise).

        Returns
        -------
        signal
            Signal template for the given parameters. An array of [pol, freq offset].
        """

        def _combine(vec):
            # Combine the bias terms and templates to get a new template
            return b_HI * b_g * vec[0] + b_HI * vec[1] + b_g * vec[2] + vec[3]

        # Generate the signal for the base model
        signal = _combine(self._stack_comp["base"][0])

        # Add in any derivative contributions
        for name, (_, x0) in self._derivs.items():

            stack = _combine(self._stack_comp[name][0])

            name = self._aliases.get(name, name)
            if name not in kwargs:
                raise ValueError(f"Need a value for deriv parameter {name}")

            x = kwargs[name]

            signal += stack * (x - x0)

        # Convolve signal with a kernel
        # before adding in the non-component contributions
        signal = self.convolve_pre_noncomp(signal, **kwargs)

        # Add in any non-component contributins
        for name, stack in self._stack_noncomp.items():

            name = self._aliases.get(name, name)
            if name not in kwargs:
                raise ValueError(f"Need a value for non-comp parameter {name}")

            x = kwargs[name]

            signal += stack[0] * x

        # Convolve signal with a kernel
        # after adding in the non-component contributions
        signal = self.convolve_post_noncomp(signal, **kwargs)

        # Scale by the overall prefactor
        signal *= omega

        return signal

    def convolve_pre_noncomp(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Override in subclass to convolve signal with kernel pre-non-components."""
        return signal

    def convolve_post_noncomp(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Override in subclass to convolve signal with kernel post-non-components."""
        return signal

    @property
    def freq(self):
        """Get the frequency separations the template is defined at."""
        return self._freq

    @property
    def params(self):
        """The names of all the parameters needed to generate the template."""
        return (
            ["omega", "b_HI", "b_g"]
            + [self._aliases.get(name, name) for name in self._stack_comp.keys()]
            + [self._aliases.get(name, name) for name in self._stack_noncomp.keys()]
        )


class SignalTemplateFoG(SignalTemplate):
    """Create signal templates from pre-simulated modes and input parameters.

    Reconvolves the stacked signal with a kernel to simulate FoG damping,
    in contrast to the SignalTemplate class that uses a linear model for
    the FoG damping.

    Parameters
    ----------
    derivs
        A dictionary of derivates expected, giving their name (key), and a tuple of the
        parameter difference used in the simulations (between the perturbed sim and the
        base values) and the fiducial value of the parameter.
    convolutions
        A dictionary of the expected convolution parameters, giving their name (key),
        and a tuple of the parameter difference used in the simulations (between the
        perturbed sim and the base values) and the fiducial value of the parameter.
    delay_range
        The lower and upper boundary of the delay in micro-seconds that will
        be used to fit for the effective scale of the base convolution kernel.
        Defaults to (0.25, 0.80) micro-seconds.
    """

    def __init__(
        self,
        derivs: Optional[Dict[str, Tuple[float, float]]] = None,
        convolutions: Optional[Dict[str, Tuple[float, float]]] = None,
        delay_range: Optional[Tuple[float, float]] = None,
        *args,
        **kwargs,
    ):

        if derivs is None:
            derivs = {
                "NL": (0.3, 1.0),
            }
        if convolutions is None:
            convolutions = {
                "FoGh": (0.2, 1.0),
                "FoGg": (0.2, 1.0),
            }
        if delay_range is None:
            delay_range = (0.25, 0.8)

        self._convolutions = convolutions
        self._delay_range = delay_range

        super().__init__(derivs=derivs, *args, **kwargs)

    def _solve_scale(
        self, base: FrequencyStackByPol, deriv: FrequencyStackByPol, alpha: float
    ) -> np.ndarray:
        """Solve for the effective scale of the FoG damping.

        Parameters
        ----------
        base
            Stacked signal from simulations with the base parameters.
        deriv
            Stacked signal from simulations with the FoG parameter perturbed.
        alpha
            The ratio of the FoG parameter for deriv relative to base.

        Returns
        -------
        scale : np.ndarray[npol,]
            The effective scale of the transfer function:
                H(\tau) = 1 / (1 + (scale * \tau)^2)
        """

        nfreq = self.freq.size
        df = np.abs(self.freq[1] - self.freq[0])
        tau = np.fft.rfftfreq(nfreq, d=df)[np.newaxis, :]
        tau2 = tau ** 2

        mu_fft_base = np.abs(np.fft.rfft(base.stack[:], nfreq, axis=-1))
        mu_fft_deriv = np.abs(np.fft.rfft(deriv.stack[:], nfreq, axis=-1))

        var_fft_base = np.sum(
            tools.invert_no_zero(base.attrs["num"] * base.weight[:]),
            axis=-1,
            keepdims=True,
        )
        var_fft_deriv = np.sum(
            tools.invert_no_zero(deriv.attrs["num"] * deriv.weight[:]),
            axis=-1,
            keepdims=True,
        )

        ratio = mu_fft_base * tools.invert_no_zero(mu_fft_deriv)
        var_ratio = ratio ** 2 * (
            var_fft_base * tools.invert_no_zero(mu_fft_base ** 2)
            + var_fft_deriv * tools.invert_no_zero(mu_fft_deriv ** 2)
        )

        y = (ratio - 1.0) * tools.invert_no_zero(alpha ** 2 - ratio)

        w = (alpha ** 2 - ratio) ** 4 * tools.invert_no_zero(
            (alpha * 2 - 1.0) ** 2 * var_ratio
        )

        w *= ((tau >= self._delay_range[0]) & (tau <= self._delay_range[1])).astype(
            np.float32
        )

        scale2 = np.sum(w * tau2 * y, axis=-1) * tools.invert_no_zero(
            np.sum(w * tau2 ** 2, axis=-1)
        )

        return np.sqrt(scale2)

    def _interpret_stacks(self, stacks: Dict[str, FrequencyStackByPol]):

        super()._interpret_stacks(stacks)

        base = stacks["11-base"]

        self._convolution_scale = {}

        for name, (delta, x0) in self._convolutions.items():

            key = f"11-{name}"

            alpha = (x0 + delta) / x0

            if key not in stacks:
                raise RuntimeError(f"Expected derivative {name} but could not load it.")

            # Determine the effective scale
            scale = self._solve_scale(base, stacks[key], alpha)
            self._convolution_scale[name] = scale

    def convolve_pre_noncomp(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Convolve the stacked signal with the relative FoG kernel.

        Parameters
        ----------
        signal : np.ndarray[npol, nfreq]
            The stacked signal before adding the non-component contributions.
        kwargs : dict
            All parameter values.

        Returns
        -------
        signal : np.ndarray[npol, nfreq]
            The input stacked signal after convolving with the relative FoG kernel.
        """

        # Figure out the size needed to perform the convolution
        nfreq = self.freq.size
        fsize = next_fast_len(nfreq)
        fslice = slice(0, nfreq)

        # Determine the delay axis
        df = np.abs(self.freq[1] - self.freq[0])
        tau = np.fft.rfftfreq(fsize, d=df)[np.newaxis, :]

        # Calculate the fft of the signal
        fft_signal = np.fft.rfft(signal, fsize, axis=-1)

        # Construct the fft of the transfer function.
        # Assumes a Lorentzian in delay space.
        fft_transfer = np.ones_like(fft_signal)

        for name, (_, x0) in self._convolutions.items():

            scale0 = self._convolution_scale[name][:, np.newaxis]

            name = self._aliases.get(name, name)
            if name not in kwargs:
                raise ValueError(f"Need a value for convolution parameter {name}")

            x = kwargs[name]

            alpha = x / x0
            scale = alpha * scale0

            fft_transfer *= (1.0 + (scale0 * tau) ** 2) / (1.0 + (scale * tau) ** 2)

        signalc = np.fft.irfft(fft_signal * fft_transfer, fsize, axis=-1)[..., fslice]

        return signalc
