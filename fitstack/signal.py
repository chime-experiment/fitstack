import logging
import re
import glob
import os
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np

from draco.core.containers import (
    ContainerBase,
    FrequencyStackByPol,
    MockFrequencyStackByPol,
)

logger = logging.getLogger(__name__)


def stack_av(stacks: List[FrequencyStackByPol]) -> FrequencyStackByPol:
    """Average a bunch of individual stacks together.

    This also calculates an estimate of the noise as it does so.
    """

    av = FrequencyStackByPol(stacks[0])

    av.stack[:] = 0.0

    sarr = [s.stack[:] for s in stacks]

    av.stack[:] = np.mean(sarr, axis=0)
    av.var = np.var(sarr, axis=0)
    av.num = len(stacks)

    return av


def stack_av_group(stack_groups: List[MockFrequencyStackByPol]) -> FrequencyStackByPol:
    """Average a bunch of stack groups together into a single stack."""

    av = FrequencyStackByPol(axes_from=stack_groups[0])

    stack_means = []
    stack_vars = []
    stack_num = 0

    for group in stack_groups:

        stacks = group.datasets["stack"][:]
        stack_means.append(stacks.mean(axis=0))
        stack_vars.append(stacks.var(axis=0))
        stack_num += stacks.shape[0]

    av.stack[:] = np.mean(stack_means, axis=0)
    av.var = np.mean(stack_vars, axis=0)
    av.num = stack_num

    return av


def load_all(paths: List[os.PathLike]) -> List[ContainerBase]:
    """Load a set of container files from disk, skipping any corrupt ones."""

    conts = []

    for p in paths:
        try:
            c = ContainerBase.from_file(p)
        except OSError:
            logger.info(f"Error loading: {p}")

        conts.append(c)

    return conts


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
    def load_from_stackfiles(cls, pattern: str, **kwargs):
        """Load the signal template from a set of stack files.

        This will load the stack files from each location and try and compile them into
        a set which can be used to generate signal templates.

        Parameters
        ----------
        pattern
            A glob pattern that isolates the base signal templates.
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

            stacks[key] = stack_av_group(load_all(stack_files))

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
                self._factor ** 2 * stack.var[:] / stack.num,
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

            name = self._aliases.get(name, name)
            if name not in kwargs:
                raise ValueError(f"Need a value for deriv parameter {name}")

            x = kwargs[name]

            signal += _combine(self._stack_comp[name][0]) * (x - x0)

        # Add in any non-component contributins
        for name, stack in self._stack_noncomp.items():

            name = self._aliases.get(name, name)
            if name not in kwargs:
                raise ValueError(f"Need a value for non-comp parameter {name}")

            x = kwargs[name]

            signal += stack[0] * x

        # Scale by the overall prefactor
        signal *= omega

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