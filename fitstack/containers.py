"""Containers for storing data products and fit results."""

import numpy as np

from draco.core.containers import *


class StackSet1D(FrequencyStackByPol):
    """Container for all data required to perform a model fit to a 1D stack."""

    _axes = ("mock",)

    _dataset_spec = {
        "mock": {
            "axes": ["mock", "pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "transfer_function": {
            "axes": ["pol", "freq"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": False,
        },
        "template": {
            "axes": ["pol", "freq"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": False,
        },
    }


class StackSet3D(Stack3D):
    """Container for all data required to perform a model fit to a 3D stack."""

    _axes = ("mock",)

    _dataset_spec = {
        "mock": {
            "axes": ["mock", "pol", "delta_ra", "delta_dec", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "transfer_function": {
            "axes": ["pol", "delta_ra", "delta_dec", "freq"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": False,
        },
        "template": {
            "axes": ["pol", "delta_ra", "delta_dec", "freq"],
            "dtype": np.float64,
            "initialise": False,
            "distributed": False,
        },
    }


class MCMCFit(ContainerBase):
    """Base container for the results of a model fit."""

    _axes = ("step", "walker", "param", "percentile")

    _dataset_spec = {
        "chain": {
            "axes": ["step", "walker", "param"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "chisq": {
            "axes": ["step", "walker"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "fixed": {
            "axes": ["param"],
            "dtype": np.bool,
            "initialise": True,
            "distributed": False,
        },
        "autocorr_time": {
            "axes": ["param"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "acceptance_fraction": {
            "axes": ["walker"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "percentile": {
            "axes": ["param", "percentile"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "median": {
            "axes": ["param"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "span_lower": {
            "axes": ["param"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "span_upper": {
            "axes": ["param"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    def samples(self, flat=True):
        """Return independent samples from the joint posterior distribution.

        Parameters
        ----------
        flat : bool
            If True, then the samples will be flattened over the
            step and walker axis.  Default is True.

        Returns
        -------
        samples : np.ndarray[nsample, nparam] or np.ndarray[nstep, nwalker, nparam]
            Samples from the joint posterior distribution after discarding
            burn-in and thinning based on the maximum autocorrelation length.
            This will be 2D if flat is True and 3D if flat is False.
        """

        samples = self.datasets["chain"][self.slice_chain, :, :]

        if flat:
            samples = samples.reshape(-1, samples.shape[-1])

        return samples

    @property
    def fit_index(self):
        return np.flatnonzero(~self.datasets["fixed"][:])

    @property
    def fixed_index(self):
        return np.flatnonzero(self.datasets["fixed"][:])

    @property
    def slice_chain(self):
        """Create a slice along the step axis that discards burn-in and thins.

        Discards 10 times the autocorrelation length and thins by the
        autocorrelation length divided by 2.
        """

        tau = int(np.max(self.datasets["autocorr_time"][:][self.fit_index]))
        thin = tau // 2
        discard = tau * 10

        return slice(discard, self.nstep, thin)

    @property
    def min_chisq(self):
        """Return the minimum chi-squared."""
        return np.min(self.datasets["chisq"][:])

    @property
    def nstep(self):
        """Return the total number of steps taken."""
        return self.index_map["step"].size


class MCMCFit1D(MCMCFit, StackSet1D):
    """Container for the results of a model fit to 1D stack and all associated data."""

    _axes = ("x",)

    _dataset_spec = {
        "cov": {
            "axes": ["pol", "pol", "freq", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "error": {
            "axes": ["pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "freq_flag": {
            "axes": ["freq"],
            "dtype": np.bool,
            "initialise": True,
            "distributed": False,
        },
        "flag": {
            "axes": ["x"],
            "dtype": np.bool,
            "initialise": True,
            "distributed": False,
        },
        "precision": {
            "axes": ["x", "x"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "model_min_chisq": {
            "axes": ["pol", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "model_percentile": {
            "axes": ["pol", "freq", "percentile"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }

    @property
    def ndof(self):
        """Return the number of degrees of freedom."""
        return np.sum(self.datasets["flag"][:]) - self.index_map["param"].size


class MCMCFit3D(MCMCFit, StackSet3D):
    """Container for the results of a model fit to 3D stack and all associated data."""

    _axes = ("pixel",)

    _dataset_spec = {
        "error": {
            "axes": ["pol", "delta_ra", "delta_dec", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "errori": {
            "axes": ["pol", "delta_ra", "delta_dec", "freq"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "cov": {
            "axes": ["pol", "pixel", "pixel"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
        "inv_cov": {
            "axes": ["pol", "pixel", "pixel"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": False,
        },
    }
