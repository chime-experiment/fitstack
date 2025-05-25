import logging
import re
import glob
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np
from scipy.fftpack import next_fast_len

from draco.util import tools
from draco.core.containers import FrequencyStackByPol, PowerSpectrum1D, PowerSpectrum2D
from draco.analysis.powerspec import get_1d_ps

from . import utils
from cora.util import cosmology
from cora.util import units as u
logger = logging.getLogger(__name__)


class SignalTemplate:
    """Create signal templates from pre-simulated modes and input parameters.

    Parameters
    ----------
    derivs
        A dictionary of derivatives expected, giving their name (key), and a tuple of the
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
        factor: float = 1.0,
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
        logger.debug(f"Using deriv modes: {self._derivs}")
        logger.debug(f"Using aliases: {self._aliases}")
        logger.debug(f"Using factor: {self._factor}")

    @classmethod
    def load_from_stackfiles(
        cls,
        pattern: str,
        pol: List[str] = None,
        weight: np.ndarray = None,
        combine: bool = True,
        sort: bool = True,
        symmetrize: bool = False,
        reverse: bool = False,
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
        symmetrize
            Explicitly symmetrize the templates.
        reverse
            Reverse the templates. Useful for testing symmetry effects.
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
            stacks[key] = utils.average_data(
                mocks, pol=mocks.pol, combine=combine, sort=sort
            )

            if reverse:
                logger.debug(f"Reversing stack {key}")
                stacks[key].stack[:] = stacks[key].stack[..., ::-1]
                stacks[key].weight[:] = stacks[key].weight[..., ::-1]

            # TODO: this presumes that 0 is the central element
            if symmetrize:
                logger.debug(f"Symmetrizing stack {key}")
                stacks[key].stack[:] = 0.5 * (
                    stacks[key].stack[:] + stacks[key].stack[..., ::-1]
                )
                stacks[key].weight[:] = 0.5 * (
                    stacks[key].weight[:] + stacks[key].weight[..., ::-1]
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
                self._factor**2
                * tools.invert_no_zero(stack.attrs["num"] * stack.weight[:]),
            )

        # For all linear component terms load them and construct the various HI,g,v
        # combination terms
        for term in compterms:
            logger.debug(f"Combining mode {term}")

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
            logger.debug(f"Interpreting derivative mode {name}")

            if name not in stack_modes:
                raise RuntimeError(f"Expected derivative {name} but could not load it.")

            s, v = stack_modes[name]
            sb, vb = stack_modes["base"]

            # Calculate the finite difference derivative
            fd_mode = (s - sb) / delta
            fd_var = (v + vb) / delta**2

            self._stack_comp[name] = (fd_mode, fd_var)

        # Load any non-component type terms. These are terms which sit outside the usual
        # bias and Kaiser factors (such as shot noise)
        noncompterms = [k for k in stacks.keys() if "-" not in k]
        for term in noncompterms:
            logger.debug(f"Interpreting non-component mode {term}")
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

        # Scale by the overall prefactor
        signal *= omega

        # Add in any non-component contributions
        for name, stack in self._stack_noncomp.items():

            name = self._aliases.get(name, name)
            if name not in kwargs:
                raise ValueError(f"Need a value for non-comp parameter {name}")

            x = kwargs[name]

            signal += stack[0] * x

        # Convolve signal with a kernel
        # after adding in the non-component contributions
        signal = self.convolve_post_noncomp(signal, **kwargs)

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
        logger.debug(f"Using convolutions: {self._convolutions}")
        logger.debug(f"Fitting delay range: {self._delay_range}")

    def _solve_scale(
        self, base: FrequencyStackByPol, deriv: FrequencyStackByPol, alpha: float
    ) -> np.ndarray:
        r"""Solve for the effective scale of the FoG damping.

        Note that the scale parameter returned by this function is different from
        the scale parameter defined in the eBOSS stacking paper: if :math:`s` is the
        code parameter and :math:`\sigma_{\rm eff}` is the paper's parameter, then

        .. math::

            s = \sigma_{\rm eff} / \sqrt{2}

        Therefore, the FoG kernel is defined as

        .. math::

            H(\tau, s) = 1 / (1 + (s \tau)^2)

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
            The effective scale of the transfer function.
        """

        nfreq = self.freq.size
        df = np.abs(self.freq[1] - self.freq[0])
        tau = np.fft.rfftfreq(nfreq, d=df)[np.newaxis, :]
        tau2 = tau**2

        # FoG kernel acts in delay space, so we FFT the stacks from freq to delay
        mu_fft_base = np.abs(np.fft.rfft(base.stack[:], nfreq, axis=-1))
        mu_fft_deriv = np.abs(np.fft.rfft(deriv.stack[:], nfreq, axis=-1))

        # Get variance of base and deriv delay-space stacks, for usage in
        # error propagation
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

        # Compute ratio of base and deriv stacks, and compute variance of
        # ratio using error propagation
        ratio = mu_fft_base * tools.invert_no_zero(mu_fft_deriv)
        var_ratio = ratio**2 * (
            var_fft_base * tools.invert_no_zero(mu_fft_base**2)
            + var_fft_deriv * tools.invert_no_zero(mu_fft_deriv**2)
        )

        # If each delay-space signal was exactly proportional to H(tau) as defined
        # in the docstring, the ratio would be equal to
        #   H(tau,alpha*s)^2 / H(tau,s)^2 .
        # This might not exactly be true because of how the data were processed,
        # but we'll assume it's true and fit for an effective value of s.
        # To do so, we write
        #   ratio = H(tau,alpha*s)^2 / H(tau,s)^2
        # and then solve for y, defined to be kpar^2 s^2.
        y = (ratio - 1.0) * tools.invert_no_zero(alpha**2 - ratio)

        # We then compute weights w that are equal to the inverse variance of y,
        # computed via error propagation. We also zero out tau values that are
        # beyond the desired fitting range
        w = (alpha**2 - ratio) ** 4 * tools.invert_no_zero(
            (alpha**2 - 1.0) ** 2 * var_ratio
        )

        w *= ((tau >= self._delay_range[0]) & (tau <= self._delay_range[1])).astype(
            np.float32
        )

        # From the definition of y, we know that s^2 = y/tau^2. We optimally
        # estimate s^2 by taking an inverse-variance weighted average of y/tau^2
        # over all tau values. (We'll only use s^2 in calculations, so it
        # makes sense to estimate s^2 instead of s.)
        scale2 = np.sum(w * tau2 * y, axis=-1) * tools.invert_no_zero(
            np.sum(w * tau2**2, axis=-1)
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

        # Loop over parameters corresponding to distinct kernels we'll need to
        # convolve the signal by
        for name, (_, x0) in self._convolutions.items():

            # Get scale corresponding to base template
            scale0 = self._convolution_scale[name][:, np.newaxis]

            # Get aliased name of parameter and parameter value
            name = self._aliases.get(name, name)
            if name not in kwargs:
                raise ValueError(f"Need a value for convolution parameter {name}")
            x = kwargs[name]

            # Re-scale effective convolution scale
            alpha = x / x0
            scale = alpha * scale0

            # Accumulate kernel into delay-space transfer function
            fft_transfer *= (1.0 + (scale0 * tau) ** 2) / (1.0 + (scale * tau) ** 2)

        # Multiply signal by transfer function and ifft back to frequency-space
        signalc = np.fft.irfft(fft_signal * fft_transfer, fsize, axis=-1)[..., fslice]

        return signalc


class AutoSignalTemplate2D:
    """Power spectrum signal templates from pre-simulated modes and input parameters.

    Parameters
    ----------
    derivs : dict
        A dictionary of derivatives expected, giving their name (key), and a tuple of the
        parameter difference used in the simulations (between the perturbed sim and the
        base values) and the fiducial value of the parameter.
    factor : float
        A scaling factor to apply to the sims.
    aliases : float
        Allow the parameters to be given by more meaningful names.
    nbins : int
        Number of 1d k bins. Default: 10.
    logbins : bool
        Whether bins should be log-spaced. Default: True.
    force_real : bool
        Force input datasets to be real. Assumes that input datasets have been previously 
        examined to verify that imaginary parts are small and/or unimportant. 
        Default: True.
    """

    def __init__(
        self,
        derivs: Optional[Dict[str, Tuple[float, float]]] = None,
        factor: float = 1.0,
        aliases: Optional[Dict[str, str]] = None,
        nbins: int = 7,
        logbins: bool = True,
        force_real: bool = True,
    ):

        if derivs is None:
            derivs = {
                "NL": (0.3, 1.0),
                "FoGh": (0.2, 1.0),
            }
        self._derivs = derivs
        self._factor = factor
        self._aliases = aliases if aliases is not None else {}
        self._nbins = nbins
        self._logbins = logbins
        self.force_real = force_real
        logger.debug(f"Using deriv modes: {self._derivs}")
        logger.debug(f"Using aliases: {self._aliases}")
        logger.debug(f"Using factor: {self._factor}")
        logger.debug(
            f"Using {self._nbins} "
            f"{'log-spaced' if self._logbins else 'linearly-spaced'} bins"
        )

    def _re(self, x):
        return np.real(x) if self.force_real else x

    @classmethod
    def load_from_ps2Dfiles(
        cls,
        pattern: str,
        filename_pattern: str = None,
        pol: List[str] = None,
        weight: np.ndarray = None,
        signal_mask: np.ndarray = None,
        combine: bool = True,
        force_real: bool = True,
        **kwargs,
    ):
        """Load the signal template from a set of 2d power spectrum files.

        This will load the ps2D files from each location and try and compile them into
        a set which can be used to generate signal templates.

        The signal templates should be stored in directories with names of the form
        `*_compderiv-d-par`, where:
        - `d` is one of `0`, `h`, or `1`, corresponding to the value of b_HI
        - `par` denotes a specific combination of other parameters

        Parameters
        ----------
        pattern
            A glob pattern that isolates the directories containing the base
            signal templates.
        filename_pattern
            A glob pattern that specifies the filenames containing the base
            signal templates.
        pol
            The desired polarisations.
        weight
            The weight to use when averaging over polarisations and binning
            from 2d to 1d. Must have shape [npol, kpara, kperp].
        signal_mask
            Boolean mask to use when binning from 2d down to 1d.
            Must have shape [npol, kpara, kperp].
        combine
            Add an element to the polarisation axis called I that
            is the weighted sum of the XX and YY polarisation.
        force_real
            Force input datasets to be real. Assumes that input datasets have 
            been previously examined to verify that imaginary parts are small 
            and/or unimportant. Default: True.
        **kwargs
            Arguments passed on to the constructor.
        """

        dirs = glob.glob(pattern)

        if filename_pattern is None:
            filename_pattern = "*.h5"

        matching = {}

        # Find directories which match the right format
        for d in sorted(dirs):
            ###mo = re.search(r"_compderiv-([^\/]+)", d)
            mo_comp = re.search(r"bias_([0-9\.]+)_Pk_([^_]+)/", d)
            # Check for shotnoise directory
            mo_shotnoise = re.search(r"template_shotnoise", d)

            if mo_comp:
                bias = mo_comp.group(1)  # This will be "0", "0.5", or "1"
                pk_type = mo_comp.group(2)  # This will be "base", "lin", "FoGh"
                # Create a composite key that identifies the templates
                key = f"{bias}-{pk_type}"

            elif mo_shotnoise:
                key = "shotnoise"
            else:
                print(f"Directory {d} does not match expected format, rejecting")
                continue

            logger.debug(f"Processing directory: {d}")

            if key in matching:
                raise ValueError(
                    "Did not find a unique set of modes at this location. "
                    "You might need to refine the pattern."
                )

            d = Path(d)

            if not d.is_dir():
                raise ValueError(
                    "Glob pattern for templates must point to directories"
                )

            matching[key] = Path(d)

        # For each directory load all the ps2D files and combine them
        ps2Ds = {}
        for key, d in matching.items():
            ps2D_files = sorted(list(d.glob(filename_pattern)))

            if len(ps2D_files) == 0:
                print("No files found at matching path.")
                continue

            mocks = utils.load_mocks(ps2D_files, pol=pol)
            ps2Ds[key] = utils.average_data(
                mocks, pol=mocks.index_map["pol"], combine=combine, sort=False
            )

        # Create the object
        self = cls(**kwargs)
        self.force_real = force_real

        # Save signal mask and ps2D weights, for later use in binning
        # 2d power spectrum to 1d
        self._signal_mask = (
            signal_mask
            if signal_mask is not None
            else next(iter(ps2Ds.values())).mask[:].copy()
        )
        self._ps2D_weight = (
            self._re(weight)
            if weight is not None
            else self._re(next(iter(ps2Ds.values())).weight[:].copy())
        )

        # Try and construct all the required templates from the stacks
        self._interpret_ps2Ds(ps2Ds)

        return self

    def _interpret_ps2Ds(
        self,
        ps2Ds: Dict[str, PowerSpectrum1D],
    ):
        # Generate the required templates from the 2d power spectra

        # Find all entries that have the linear component structure
        compterms = [k.split("-")[1] for k in ps2Ds.keys() if k.startswith("0-")]

        ps2D_modes = {}

        # Get the first kpara, kperp axes as references
        self._kpara = next(iter(ps2Ds.values())).kpara[:].copy()
        self._kperp = next(iter(ps2Ds.values())).kperp[:].copy()
        self._kpara.flags.writeable = False
        self._kperp.flags.writeable = False

        def _check_load_ps2D(key):
            # Validate the 2D power spectrum and extract the template and its variance

            if key not in ps2Ds:
                raise RuntimeError(f"Power spectrum {key} was not loaded.")

            ps2D = ps2Ds[key]

            if not np.array_equal(ps2D.kpara[:], self._kpara):
                raise RuntimeError(
                    f"k_par values in power spectrum {key} do not match reference."
                )

            if not np.array_equal(ps2D.kperp[:], self._kperp):
                raise RuntimeError(
                    f"k_perp values in power spectrum {key} do not match reference."
                )

            return (
                self._factor * self._re(ps2D.spectrum[:]),
                self._factor**2
                * tools.invert_no_zero(ps2D.attrs["num"] * self._re(ps2D.weight[:])),
            )

        # For all linear component terms, load them and construct the various HI,v
        # combination terms
        for term in compterms:
            logger.debug(f"Combining mode {term}")

            s0, v0 = _check_load_ps2D(f"0-{term}")
            sh, vh = _check_load_ps2D(f"0.5-{term}")
            s1, v1 = _check_load_ps2D(f"1-{term}")

            # Initialize arrays for b_HI = 0, 1/2, 1
            template_mean = np.zeros((3,) + s0.shape)
            template_var = np.zeros((3,) + s0.shape)

            # Calculate the template for each component
            ## s_hh = 2 [s(1,1,0) - 2s(1,1/2,0) + s(1,0,0)]
            template_mean[0] = 2 * (s1 - 2 * sh + s0)
            ## s_hv = s(1,1/2,0) - s(1,0,0) - 1/4 shh
            template_mean[1] = sh - s0 - 0.25 * template_mean[0]
            ## s_vv = s(1,0,0)
            template_mean[2] = s0

            # Calculate the variance of each component, using error propagation
            template_var[0] = 4 * (v1 + 4 * vh + v0)
            template_var[1] = vh + v0 + 0.0625 * template_var[0]
            template_var[2] = v0

            ps2D_modes[term] = (template_mean, template_var)

        self._ps2D_comp = {}
        self._ps2D_noncomp = {}
        self._ps2D_comp["base"] = ps2D_modes["base"]

        # For the expected derivative modes, combine the perturbed entry and the base
        # templates to get the derivative templates
        for name, (delta, _) in self._derivs.items():
            logger.debug(f"Interpreting derivative mode {name}")

            if name not in ps2D_modes:
                raise RuntimeError(f"Expected derivative {name} but could not load it.")

            s, v = ps2D_modes[name]
            sb, vb = ps2D_modes["base"]

            # Calculate the finite difference derivative
            fd_mode = (s - sb) / delta
            fd_var = (v + vb) / delta**2

            self._ps2D_comp[name] = (fd_mode, fd_var)

        # Load any non-component type terms. These are terms which sit outside the usual
        # bias and Kaiser factors (such as shot noise)
        noncompterms = [key for key in ps2Ds.keys() if "-" not in key]
        for term in noncompterms:
            logger.debug(f"Interpreting non-component mode {term}")
            self._ps2D_noncomp[term] = _check_load_ps2D(term)

    def signal_2D(self, *, omega: float, b_HI: float, **kwargs: float) -> np.ndarray:
        """Return the 2D power spectrum signal template for the given parameters.

        Parameters
        ----------
        omega
            Overall scaling.
        b_HI
            Scaling for the HI bias term.
        **kwargs
            Values for all other derivative terms (e.g. NL) and non-component terms
            (e.g. shotnoise).

        Returns
        -------
        signal
            Signal template for the given parameters. An array of [pol, kpara, kperp].
        """

        def _combine(vec):
            # Combine the bias terms and templates to get a new template
            return b_HI**2 * vec[0] + 2 * b_HI * vec[1] + vec[2]

        # Generate the signal for the base model
        signal = _combine(self._ps2D_comp["base"][0])

        # Add in any derivative contributions
        for name, (_, x0) in self._derivs.items():

            ps2D = _combine(self._ps2D_comp[name][0])

            name = self._aliases.get(name, name)
            if name not in kwargs:
                raise ValueError(f"Need a value for deriv parameter {name}")

            x = kwargs[name]

            signal += ps2D * (x - x0)

        # Multiply signal by a Fourier-space function
        # before adding in the non-component contributions
        signal = self.multiply_pre_noncomp(signal, **kwargs)

        # Scale by the overall prefactor (omega**2 for auto-correlation)
        signal *= omega**2

        # Add in any non-component contributions
        for name, ps2D in self._ps2D_noncomp.items():

            name = self._aliases.get(name, name)
            if name not in kwargs:
                raise ValueError(f"Need a value for non-comp parameter {name}")

            x = kwargs[name]

            signal += ps2D[0] * x

        # Multiply signal by a Fourier-space function
        # after adding in the non-component contributions
        signal = self.multiply_post_noncomp(signal, **kwargs)

        return signal

    def signal_1D(self, *, omega: float, b_HI: float, **kwargs: float) -> np.ndarray:
        """Return the 1D power spectrum template, binned from 2D template.

        Parameters
        ----------
        omega
            Overall scaling.
        b_HI
            Scaling for the HI bias term.
        **kwargs
            Values for all other derivative terms (e.g. NL) and non-component terms
            (e.g. shotnoise).

        Returns
        -------
        signal
            Signal template for the given parameters. An array of [pol, k].
        """

        _signal_2D = self.signal_2D(omega=omega, b_HI=b_HI, **kwargs)

        signal_1D = np.zeros((_signal_2D.shape[0], self._nbins))

        for ipol in range(_signal_2D.shape[0]):

            _, signal_1D[ipol], _, _, _ = get_1d_ps(
                _signal_2D[ipol],
                self._kperp,
                self._kpara,
                self._ps2D_weight[ipol],
                self._signal_mask[ipol],
                self._nbins + 1,
                self._logbins,
            )

        return signal_1D

    def multiply_pre_noncomp(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Override in subclass to multiply signal by function pre-non-components."""
        return signal

    def multiply_post_noncomp(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Override in subclass to multiply signal by function post-non-components."""
        return signal

    @property
    def kpara(self):
        """Get k_para values the template is defined at."""
        return self._kpara

    @property
    def kperp(self):
        """Get k_perp values the template is defined at."""
        return self._kperp

    @property
    def params(self):
        """The names of all the parameters needed to generate the template."""
        return (
            ["omega", "b_HI"]
            + [self._aliases.get(name, name) for name in self._ps2D_comp.keys()]
            + [self._aliases.get(name, name) for name in self._ps2D_noncomp.keys()]
        )


class AutoSignalTemplate2DFoG(AutoSignalTemplate2D):
    """Create signal templates from pre-simulated modes and input parameters.

    Multiplies the 2d power spectrum with a kernel to simulate FoG damping,
    in contrast to the AutoSignalTemplate2D class, which uses a linear model for
    the FoG damping.

    Parameters
    ----------
    derivs
        A dictionary of derivatives expected, giving their name (key), and a tuple of the
        parameter difference used in the simulations (between the perturbed sim and the
        base values) and the fiducial value of the parameter.
    convolutions
        A dictionary of the expected convolution parameters, giving their name (key),
        and a tuple of the parameter difference used in the simulations (between the
        perturbed sim and the base values) and the fiducial value of the parameter.
    kpara_range
        The lower and upper boundary of k_parallel that will be used to fit for
        the effective scale of the base convolution kernel.
        Defaults to (0, 5) Mpc^-1.
    """

    def __init__(
        self,
        derivs: Optional[Dict[str, Tuple[float, float]]] = None,
        convolutions: Optional[Dict[str, Tuple[float, float]]] = None,
        kpara_range: Optional[Tuple[float, float]] = None,
        z_eff: Optional[float] = None,  # effective redshift      
        *args,
        **kwargs,
    ):
        
        # Set default z_eff if None
        self.z_eff = z_eff if z_eff is not None else 1.0
        cosmo_cora = cosmology.Cosmology()
        self.H_z = cosmo_cora.H(self.z_eff) * u.mega_parsec / 1000.  # In km/s/Mpc 

        if derivs is None:
            derivs = {
                "NL": (0.3, 1.0),
            }
        if convolutions is None:
            convolutions = {
                "FoGh": (0.2, 1.0),
            }
        if kpara_range is None:
            kpara_range = (0.0, 5.0)

        self._convolutions = convolutions
        self._kpara_range = kpara_range

        super().__init__(derivs=derivs, *args, **kwargs)
        logger.debug(f"Using convolution parameters: {self._convolutions}")
        logger.debug(
            f"Fitting effective FoG scale over k_para range: {self._kpara_range}"
        )

    def _solve_scale(
        self, base: PowerSpectrum2D, deriv: PowerSpectrum2D, alpha: float
    ) -> np.ndarray:
        r"""Solve for the effective scale of the FoG damping.

        Note that the scale parameter returned by this function is different from
        the scale parameter defined in the eBOSS stacking paper: if :math:`s` is the
        code parameter and :math:`\sigma_{\rm eff}` is the paper's parameter, then

        .. math::

            s = \sigma_{\rm eff} / \sqrt{2}

        Therefore, the FoG kernel is defined as

        .. math::

            H(k_\parallel, s) = 1 / (1 + (s k_\parallel)^2)

        Parameters
        ----------
        base
            2d power spectrum from simulations with the base parameters.
        deriv
            2d power spectrum from simulations with the FoG parameter perturbed.
        alpha
            The ratio of the FoG parameter for deriv relative to base.

        Returns
        -------
        scale : np.ndarray[npol,]
            The effective scale of the transfer function.
        """

        kpara2 = self.kpara[np.newaxis, :, np.newaxis] ** 2

        # Take real parts of input spectra, so that output scale is also real
        ps2D_base = self._re(base.spectrum[:])
        ps2D_deriv = self._re(deriv.spectrum[:])

        # Get variance of base and deriv ps2D, for usage in error propagation
        var_ps2D_base = tools.invert_no_zero(self._re(base.weight[:]))
        var_ps2D_deriv = tools.invert_no_zero(self._re(deriv.weight[:]))

        # Compute ratio of base and deriv ps2D, and compute variance in ratio using
        # error propagation
        ratio = ps2D_base / ps2D_deriv
        var_ratio = ratio**2 * (
            var_ps2D_base * tools.invert_no_zero(ps2D_base**2)
            + var_ps2D_deriv * tools.invert_no_zero(ps2D_deriv**2)
        )

        # If each power spectrum was exactly proportional to H(kpar) as defined in
        # the docstring, the ratio would be equal to
        #   H(kpar,alpha*s)^2 / H(kpar,s)^2 .
        # This might not exactly be true because of how the data were processed,
        # but we'll assume it's true and fit for an effective value of s.
        # To do so, we write
        #   ratio = H(kpar,alpha*s)^2 / H(kpar,s)^2
        # and then solve for y, defined to be kpar^2 s^2.
        r_sqrt = ratio**0.5
        y = (r_sqrt - 1.0) * tools.invert_no_zero(alpha**2 - r_sqrt)

        # We then compute weights w that are equal to the inverse variance of y,
        # computed via error propagation. We also zero out kpar values that are
        # beyond the desired fitting range
        w = (
            4
            * ratio
            * (alpha**2 - r_sqrt) ** 4
            * tools.invert_no_zero((alpha * 2 - 1.0) ** 2 * var_ratio)
        )
        w_mask = (self.kpara >= self._kpara_range[0]) & (self.kpara <= self._kpara_range[1])
        w *= w_mask[np.newaxis, :, np.newaxis]

        # From the definition of y, we know that s^2 = y/kpar^2. We optimally
        # estimate s^2 by taking an inverse-variance weighted average of y/kpar^2
        # over all (kpar,kperp) values. (We'll only use s^2 in calculations, so it
        # makes sense to estimate s^2 instead of s.)
        scale2 = np.sum(w * kpara2 * y, axis=(-1, -2)) * tools.invert_no_zero(
            np.sum(w * kpara2**2, axis=(-1, -2))
        )

        return np.sqrt(scale2)

    def _interpret_ps2Ds(self, ps2Ds: Dict[str, PowerSpectrum1D]):

        super()._interpret_ps2Ds(ps2Ds)

        base = ps2Ds["1-base"]

        self._convolution_scale = {}

        for name, (delta, x0) in self._convolutions.items():

            key = f"1-{name}"

            alpha = (x0 + delta) / x0

            if key not in ps2Ds:
                raise RuntimeError(f"Expected derivative {name} but could not load it.")

            # Determine the effective scale
            scale = self._solve_scale(base, ps2Ds[key], alpha)
            self._convolution_scale[name] = scale


    def _get_factor(self) -> float:
        """Calculate the conversion factor connecting tau and k_parallel.
        
        C = -1/(2 pi nu21) * c/H(z) * (1+z)²
        
        Returns
        -------
        C : float
            Conversion factor
        """

        C = (-1.0 / (2 * np.pi * u.nu21)) * ((u.c / u.kilo) / self.H_z) * (1 + self.z_eff)**2

        return C            

    def multiply_pre_noncomp(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Multiply the 2d power spectrum with the relative FoG kernel.

        Parameters
        ----------
        signal : np.ndarray[npol, nkpara, nkperp]
            The 2d power spectrum before adding the non-component contributions.
        kwargs : dict
            All parameter values.

        Returns
        -------
        signal : np.ndarray[npol, nkpara, nkperp]
            The 2d power spectrum after multiplication with the relative FoG kernel.
        """
        # Calculate the conversion factor
        C = self._get_factor()
        
        # Loop over parameters corresponding to distinct kernels we'll need to
        # multiply into the signal
        for name, (_, x0) in self._convolutions.items():

            # Get scale corresponding to base template
            scale0 = self._convolution_scale[name][:, np.newaxis, np.newaxis]

            # Get aliased name of parameter and parameter value
            name = self._aliases.get(name, name)
            if name not in kwargs:
                raise ValueError(f"Need a value for convolution parameter {name}")
            x = kwargs[name]

            # Re-scale effective convolution scale
            alpha = x / x0
            scale = alpha * scale0

            # Multiply kernel into signal
            signal *= (1.0 + (scale0 * C * self.kpara[np.newaxis, :, np.newaxis]) ** 2) / (
                1.0 + (scale * C * self.kpara[np.newaxis, :, np.newaxis]) ** 2
            )

        return signal
