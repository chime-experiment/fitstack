import logging
import inspect

import numpy as np
import scipy.optimize
import scipy.stats

from caput import config

from draco.core import task

from . import containers
from . import utils
from . import models
from . import priors

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from tqdm.notebook import trange

    TQDM_IMPORTED = True
except ImportError:
    logger.warning("Error importing tqdm")
    TQDM_IMPORTED = False


BOUNDED_MINIMIZATION = ["L-BFGS-B", "Nelder-Mead", "Powell", "TNC"]


def get_param0(fit, param_to_fit):
    """Get best-fit parameters from MCMC results.

    The intent is that the parameter values from this routine can
    be used as an initial guess for a direct chi^2-minimization
    procedure.

    Parameters
    ----------
    fit : containers.MCMCFit
        Container containing MCMC results.
    param_to_fit : list of str
        List of parameter names to retrieve.

    Returns
    -------
    theta_min : np.ndarray
        Array of parameter values matching names in param_to_fit.
    """

    # Get chi^2 and parameter values at each sample from chain,
    # along with parameter names
    chisq = fit["chisq"][:]
    chain = fit["chain"][:]
    param = list(fit.index_map["param"][:])

    # Find index of chain sample with smallest chi^2, then get
    # corresponding parameter values
    imin = np.argmin(np.abs(chisq.flatten()))
    theta_min = chain.reshape(-1, chain.shape[-1])[imin]

    # Identify mapping from array of desired parameters to ordering
    # of parameters in chain results
    ifit = np.array([param.index(pfit) for pfit in param_to_fit])

    # Return correctly-ordered parameter values
    return theta_min[ifit]


def get_bounds(mdl, scale_bound=0.0):
    """Get parameter bounds to use in chi^2 minimization.

    Parameters
    ----------
    mdl : models.Model
        Container with model information.
    scale_bound : float, optional
        If nonzero, expand parameter bounds by this factor.
        Default: 0.
    """

    # Get parameter names
    param = mdl.param_name_fit
    nparam = len(param)

    # Make empty arrays to hold lower and upper bounds for each parameter
    lb = np.zeros(nparam, dtype=np.float64)
    ub = np.zeros(nparam, dtype=np.float64)

    # Loop through parameters
    for nn, name in enumerate(param):

        # Set parameter bounds based on whether prior is uniform or Gaussian
        prior = mdl.priors[name]
        if isinstance(prior, (priors.Uniform, priors.PowerLaw)):
            lb[nn] = prior.low
            ub[nn] = prior.high
        else:
            lb[nn] = prior.loc - 5.0 * prior.scale
            ub[nn] = prior.loc + 5.0 * prior.scale

        # Expand bounds if desired
        if scale_bound > 0.0:
            db = ub[nn] - lb[nn]
            lb[nn], ub[nn] = lb[nn] - scale_bound * db, ub[nn] + scale_bound * db

    return scipy.optimize.Bounds(lb, ub)


def powerspectrum1d_min_chisq_fit(
    mcmcfit_cont,
    model_kwargs=None,
    param_spec=None,
    param0=None,
    method="L-BFGS-B",
    options=None,
    scale_bound=0.0,
    force_real=True,
    add_mock_to_data=None,
    save_bestfit_models=False,
    verbose_notebook=False,
):
    """Compute the minimum chi^2 for 1d power spectrum data and mocks.

    Parameters
    ----------
    mcmc_fit_cont : containers.MCMCFitPowerSpectrum1D or str
        Container (or container filename) with information about data,
        mocks, covariance, and signal model.
    model_kwargs : dict, optional
        Dictionary that contains any keyword arguments that should be passed
        to the model class at initialization. Arguments used to generate
        `mcmc_fit_cont` will be inherited, unless explicitly overridden.
        Default: None.
    param_spec : dict, optional
        Dictionary that specifies the prior distribution for each parameter.
        See the docstring for the models.Model attribute for the correct format.
        Arguments used to generate `mcmc_fit_cont` will be inherited, unless explicitly
        overridden. Default: None.
    param0 : array_like, optional
        Starting guess for parameter values. If not set, determined from input chain.
        Default: None
    method : str, optional
        Method for `scipy.optimize.minimize`. Default: L-BFGS-B.
    options : dict, optional
        Dictionary of options for `scipy.optimize.minimize`, Default: None.
    scale_bound : float, optional
        Scale allowed bounds for parameters by this factor. Default: 0.0.
    force_real : bool, optional
        Force input datasets to be real. Assumes that input datasets have
        been previously examined to verify that imaginary parts are small and/or
        unimportant. Default: True.
    add_mock_to_data: int, optional
        Add mock with this index to data before performing fit. This is intended
        as a quick way to add a noise realization to an input signal-only
        simulation. Default: None.
    save_bestfit_models : bool, optional
        Whether to save best-fit model evaluations for data and each mock, as
        datasets in output container. Default: False.
    verbose_notebook : bool, optional
        Whether to print status updates when evaluating in a jupyter notebook,
        using the `tqdm` package. Ignored if `tqdm` is not installed.
        Default: False.

    Returns
    -------
    out : containers.ChisqPowerSpectrum1D
        Container containing chi^2 results and associated information.
    """

    def _re(x):
        return np.real(x) if force_real else x

    if model_kwargs is None:
        model_kwargs = {}

    if param_spec is None:
        param_spec = {}

    if options is None:
        options = {}

    # Load MCMCFitPowerSpectrum1D container
    if isinstance(mcmcfit_cont, str):
        fit_cont = containers.MCMCFitPowerSpectrum1D.from_file(
            utils.find_file(mcmcfit_cont)
        )
    else:
        fit_cont = mcmcfit_cont

    # Get polarizations from input container
    pol = fit_cont.index_map["pol"]
    pol_fit = fit_cont.attrs["pol_fit"]
    if pol_fit == "joint":
        ipol = np.arange(len(pol))
    else:
        ipol = list(pol).index(fit_cont.attrs["pol_fit"])
        if isinstance(ipol, int):
            ipol = [ipol]

    # Set quantities needed for model evaluation
    model_kwargs["combine"] = False
    input_model_kwargs = fit_cont.attrs["model_kwargs"]
    for key in input_model_kwargs.keys():
        if key not in model_kwargs.keys():
            model_kwargs[key] = input_model_kwargs[key]

    # Set quantities needed for fit to data
    fit_kwargs = {}
    fit_kwargs["k1D"] = fit_cont.k1D
    fit_kwargs["data"] = _re(fit_cont.spectrum.data.local_array[ipol])
    fit_kwargs["inv_cov"] = fit_cont["precision"][:]
    fit_kwargs["transfer"] = None
    fit_kwargs["pol_sel"] = ipol

    # Set parameter priors
    input_param_spec = fit_cont.attrs["param_spec"]
    for key in input_param_spec.keys():
        if key not in param_spec.keys():
            param_spec[key] = input_param_spec[key]

    # Initialize signal model, based on name stored in
    # MCMCFitPowerSpectrum1D container
    signal_model_name = fit_cont.attrs["model"]
    signal_model_class = getattr(models, signal_model_name)
    signal_model = signal_model_class(**{**model_kwargs, **param_spec})
    signal_model.set_data(**fit_kwargs)

    # Initialize null model
    null_model = models.NullModel(**model_kwargs)
    null_model.set_data(**fit_kwargs)

    # Get mocks from MCMCFitPowerSpectrum1D container
    mock_data = fit_cont["mock"][:, ipol]
    nmock, npol, nk = mock_data.shape

    # If desired, add one of the mocks to the data
    if add_mock_to_data is not None:
        fit_kwargs["data"][:] += mock_data[add_mock_to_data][:]
        signal_model.set_data(**fit_kwargs)

    # Create container for results
    out = containers.ChisqPowerSpectrum1D(
        mock=np.arange(nmock, dtype=int),
        param=np.array(signal_model.param_name_fit),
        pol=np.array(pol)[ipol],
        k=nk,
    )

    # Determine initial guess for parameter values, and parameter bounds (if needed)
    if param0 is None:
        param0 = get_param0(mcmcfit_cont, signal_model.param_name_fit)
    if method in BOUNDED_MINIMIZATION:
        param_bounds = get_bounds(signal_model, scale_bound=scale_bound)
    else:
        param_bounds = None

    # Minimize negative log-likelihood for fitting signal model to data
    if signal_model.nfit == 1:
        resd = scipy.optimize.minimize_scalar(
            signal_model.negative_log_likelihood,
            param0,
            method="bounded",
            bounds=[param_bounds.lb, param_bounds.ub],
            options=options,
        )
    else:
        resd = scipy.optimize.minimize(
            signal_model.negative_log_likelihood,
            param0,
            method=method,
            bounds=param_bounds,
            options=options,
        )

    # Save results, along with data itself, to output container
    out.attrs["data_signal_success"] = resd.success
    out.attrs["data_signal_chisq"] = 2.0 * signal_model.negative_log_likelihood(resd.x)
    out.attrs["data_signal_bestfit_param"] = resd.x
    out.attrs["data"] = fit_kwargs["data"][:]

    # Save chi^2 for null model and data
    out.attrs["data_null_chisq"] = 2.0 * null_model.negative_log_likelihood([])

    # Dereference datasets for mock fitting
    success = out["success"][:].view(np.ndarray)
    chisq_null = out["chisq_null"][:].view(np.ndarray)
    chisq_signal = out["chisq_signal"][:].view(np.ndarray)
    bestfit_param = out["bestfit_param"][:].view(np.ndarray)

    if save_bestfit_models:
        # Save best-fit model prediction for data
        out.add_dataset("data_bestfit_model")
        out.datasets["data_bestfit_model"][:] = signal_model.model(
            signal_model.get_all_params(resd.x)
        )

        # Initialize dataset to store best-fit model predictions for mocks
        out.add_dataset("mock_bestfit_models")

    # Loop over mocks
    if verbose_notebook and TQDM_IMPORTED:
        mock_iter = trange(nmock)
    else:
        mock_iter = range(nmock)

    for mm in mock_iter:

        # Print progress update to log
        if (mm % 100) == 0:
            logger.info(f"Fitting data realization {mm} of {nmock}.")

        # Update models to consider mock data
        fit_kwargs["data"] = _re(mock_data[mm])
        signal_model.set_data(**fit_kwargs)
        null_model.set_data(**fit_kwargs)

        # Minimize negative log-likelihood for fitting signal model to mock data,
        # and save results
        if signal_model.nfit == 1:
            resd = scipy.optimize.minimize_scalar(
                signal_model.negative_log_likelihood,
                param0,
                method="bounded",
                bounds=[param_bounds.lb, param_bounds.ub],
                options=options,
            )
        else:
            resd = scipy.optimize.minimize(
                signal_model.negative_log_likelihood,
                param0,
                method=method,
                bounds=param_bounds,
                options=options,
            )
        success[mm] = resd.success
        chisq_signal[mm] = 2.0 * signal_model.negative_log_likelihood(resd.x)
        bestfit_param[mm] = resd.x

        if save_bestfit_models:
            out.datasets["mock_bestfit_models"][mm] = signal_model.model(
                signal_model.get_all_params(resd.x)
            )

        # Save chi^2 for null model and data
        chisq_null[mm] = 2.0 * null_model.negative_log_likelihood([])

    # Fit a chi^2 distribution with an unknown number of d.o.f. to the
    # Delta chi^2 values for each mock
    dchisq_mocks = chisq_null[success] - chisq_signal[success]
    ndof_mocks = scipy.stats.chi2.fit(dchisq_mocks, floc=0, fscale=1)[0]
    out.attrs["dchisq_distribution_ndof"] = ndof_mocks

    # Compute p-value and "number of sigmas" for Delta chi^2 value for data
    dchisq_data = out.attrs["data_null_chisq"] - out.attrs["data_signal_chisq"]
    data_pvalue = 1 - scipy.stats.chi2.cdf(dchisq_data, ndof_mocks)
    data_nsigmas = scipy.stats.norm.ppf(1 - data_pvalue)

    out.attrs["data_pvalue"] = data_pvalue
    out.attrs["data_nsigmas"] = data_nsigmas

    # Return the output container
    return out


class PowerSpectrum1DMinChisqFit(task.SingleTask):
    """Pipeline task that calls the powerspectrum1d_min_chisq_fit function.

    Enables the user to call the powerspectrum1d_min_chisq_fit method
    with caput-pipeline, which provides many useful features including
    profiling, job script generation, job templating, and saving the
    results to disk.

    See the arguments of the powerspectrum1d_min_chisq_fit method for a
    list of attributes and their default values.
    """

    model_kwargs = config.Property(proptype=dict)
    param_spec = config.Property(proptype=dict)
    param0 = config.Property(proptype=list)
    method = config.Property(proptype=str)
    options = config.Property(proptype=dict)
    scale_bound = config.Property(proptype=float)
    force_real = config.Property(proptype=bool)
    add_mock_to_data = config.Property(proptype=int)
    save_bestfit_models = config.Property(proptype=bool)

    def setup(self):
        """Prepare all arguments for the powerspectrum1d_min_chisq_fit method."""

        # Use the default values from the powerspectrum1d_min_chisq_fit method,
        # so we do not have to repeat them in two places.
        signature = inspect.signature(powerspectrum1d_min_chisq_fit)
        defaults = {
            k: v.default if v.default is not inspect.Parameter.empty else None
            for k, v in signature.parameters.items()
        }

        self.kwargs = {}
        for key, default_val in defaults.items():
            if hasattr(self, key):
                prop_val = getattr(self, key)
                self.kwargs[key] = prop_val if prop_val is not None else default_val
            else:
                self.log.warning(
                    "PowerSpectrum1DMinChisqFit does not have a property "
                    f"corresponding to the {key} keyword argument to "
                    "powerspectrum1d_min_chisq_fit."
                )

    def process(self, mcmcfit_cont):
        """Run the chi^2 minimization.

        Parameters
        ----------
        mcmc_fit_cont : containers.MCMCFitPowerSpectrum1D
            Container with information about data, mocks, covariance, and
            signal model.

        Returns
        -------
        out : containers.ChisqPowerSpectrum1D
            Container containing chi^2 results and associated information.
        """
        out = powerspectrum1d_min_chisq_fit(
            mcmcfit_cont, verbose_notebook=False, **self.kwargs
        )

        return out
