import logging
import inspect

import numpy as np
import scipy.optimize
import scipy.stats

from caput import config, mpiarray

from draco.core import task

from . import containers
from . import utils
from . import models
from . import priors
from . import stats

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


def get_powerspectrum1d_LOO_invcovariance(mocks, iLOO, fit_cont, hartlap=True):
    """Compute leave-one-out inverse covariance from mocks.

    This routine computes the inverse covariance from all mocks
    except the one specified by index iLOO.

    Parameters
    ----------
    mocks : np.ndarray[nmock, npol, nk]
        Array of mocks
    iLOO : int
        Index of mock to leave out.
    fit_cont : containers.MCMCFitPowerSpectrum1D
        Container that mocks came from, which other metadata will
        be drawn from.
    hartlap : bool, optional
        Apply the Hartlap factor to the inverse covariance.
        Default: True

    Returns
    -------
    Cinv : np.ndarray[npol*nk, npol*nk]
        Inverse covariance.
    hartlap_factor : float
        Hartlap factor applied to inverse covariance (1.0 if factor
        not applied).
    """

    nmock, npol, nx = mocks.shape
    nmock -= 1

    # Get quantities for constructing inverse covariance
    ifit = fit_cont.attrs["ifit"]
    flag_before = fit_cont.attrs["flag_before"]

    # Compute covariance. (Relevant pols were already selected in
    # mocks, so no need to sub-select pols here)
    C = utils.covariance(np.delete(mocks, iLOO, axis=0).reshape(nmock, -1), corr=False)
    if flag_before:
        C = C[ifit][:, ifit]

    # Invert the covariance matrix to obtain the precision matrix
    Cinvfit = np.linalg.pinv(C)

    # Compute and apply the Hartlap factor to the inverse covariance
    if hartlap:
        hartlap_factor = (nmock - Cinvfit.shape[0] - 2.0) / (nmock - 1.0)
        Cinvfit *= hartlap_factor
    else:
        hartlap_factor = 1.0

    if not flag_before:
        Cinvfit = Cinvfit[ifit][:, ifit]

    Cinv = np.zeros_like(C)
    for ii, oi in enumerate(ifit):
        Cinv[oi, ifit] = Cinvfit[ii, :]

    return Cinv, hartlap_factor


def initialize_powerspectrum1d_min_chisq_ingredients(
    mcmcfit_cont,
    mcmcfit_cont_for_mocks=None,
    model_kwargs=None,
    param_spec=None,
    param0=None,
    extra_starts=0,
    extra_start_log_bounds=None,
    extra_start_log_samplenegative=True,
    bounded=True,
    scale_bound=0.0,
    force_real=True,
    add_mock_to_data=None,
    initialize_with_mock=None,
    seed=0,
):
    """Prepare parameters and data structures for chi^2 minimization.

    See docstring of `powerspectrum1d_min_chisq_fit()` for parameter
    definitions, except `initialize_with_mock`, which is either an
    integer denoting the index of a mock with which to initialize
    the signal model, or `None` if the model is to be initialized
    with the data.

    Returns
    -------
    signal_model : models.Model
        Signal model.
    null_model : models.NullModel
        No-signal model.
    mock_data : np.ndarray[nmock,npol,nk]
        Array of noise mocks.
    fit_kwargs : dict
        kwargs for signal model.
    param0_points : np.ndarray[npoint,nparam]
        Array of random starting points in parameter space.
    param_bounds : scipy.optimize.Bounds
        Bounds for each parameter.
    fit_cont_for_mocks : containers.MCMCFitPowerSpectrum1D
        Container containing mocks.
    out : containers.ChisqPowerSpectrum1D
        Container to store chi^2 results and associated information.
    """

    def _re(x):
        return np.real(x) if force_real else x

    if model_kwargs is None:
        model_kwargs = {}

    if param_spec is None:
        param_spec = {}

    # Load MCMCFitPowerSpectrum1D container
    if isinstance(mcmcfit_cont, str):
        fit_cont = containers.MCMCFitPowerSpectrum1D.from_file(
            utils.find_file(mcmcfit_cont)
        )
    else:
        fit_cont = mcmcfit_cont

    # Load separate MCMCFitPowerSpectrum1D container with mocks, if specified
    if mcmcfit_cont_for_mocks is None:
        fit_cont_for_mocks = fit_cont
    else:
        if isinstance(mcmcfit_cont_for_mocks, str):
            fit_cont_for_mocks = containers.MCMCFitPowerSpectrum1D.from_file(
                utils.find_file(mcmcfit_cont_for_mocks)
            )
        else:
            fit_cont_for_mocks = mcmcfit_cont_for_mocks

    # Get polarizations from input container
    pol = fit_cont.index_map["pol"]
    pol_fit = fit_cont.attrs["pol_fit"]
    if pol_fit == "joint":
        ipol = np.arange(len(pol))
    else:
        ipol = list(pol).index(fit_cont.attrs["pol_fit"])
        if isinstance(ipol, int):
            ipol = [ipol]

    # Get mocks from MCMCFitPowerSpectrum1D container
    mock_data = fit_cont_for_mocks["mock"][:, ipol]
    nmock, _, nk = mock_data.shape

    # Set quantities needed for model evaluation
    model_kwargs["combine"] = False
    input_model_kwargs = fit_cont.attrs["model_kwargs"]
    for key in input_model_kwargs.keys():
        if key not in model_kwargs.keys():
            model_kwargs[key] = input_model_kwargs[key]

    # Set quantities needed for fit to data
    fit_kwargs = {}
    fit_kwargs["k1D"] = fit_cont.k1D
    fit_kwargs["inv_cov"] = fit_cont["precision"][:]
    fit_kwargs["transfer"] = None
    fit_kwargs["pol_sel"] = ipol

    # Set data to fit to
    if initialize_with_mock is None:
        fit_kwargs["data"] = (
            _re(fit_cont.spectrum.data.local_array[ipol])
            if type(fit_cont.spectrum.data) is mpiarray.MPIArray
            else _re(fit_cont.spectrum.data[ipol])
        )
        # If desired, add one of the mocks to the data
        if add_mock_to_data is not None:
            fit_kwargs["data"][:] += mock_data[add_mock_to_data][:]
    else:
        fit_kwargs["data"] = _re(mock_data[initialize_with_mock])

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

    # Determine initial guess for parameter values, and parameter bounds (if needed)
    if param0 is None:
        param0 = get_param0(fit_cont, signal_model.param_name_fit)
    param_bounds = get_bounds(signal_model, scale_bound=scale_bound)

    # Determine additional starting points for optimizer, based on Latin hypercube
    # sampling of the parameter space
    param0_points = [param0]
    if extra_starts > 0:
        # Initialize Latin hypercube sampler
        sampler = scipy.stats.qmc.LatinHypercube(d=len(param0), seed=seed)
        # Draw extra_starts d-dimensional samples from the unit Latin hypercube
        other_param0 = sampler.random(extra_starts)
        # Scale the samples to cover the desired parameter bounds
        if extra_start_log_bounds is None:
            other_param0 = scipy.stats.qmc.scale(
                other_param0, param_bounds.lb, param_bounds.ub
            )
        else:
            extra_start_log_bounds = np.asarray(extra_start_log_bounds)
            if extra_start_log_bounds.shape != (len(param0), 2):
                raise RuntimeError(
                    "extra_start_log_bounds must have shape (nparam, 2))"
                )
            other_param0 = scipy.stats.qmc.scale(
                other_param0,
                extra_start_log_bounds[:, 0],
                extra_start_log_bounds[:, 1],
            )
            other_param0 = 10.0**other_param0
            if extra_start_log_samplenegative:
                rng = np.random.default_rng(seed=seed)
                other_param0 *= rng.choice([1.0, -1.0], size=other_param0.shape)

        # Make a list of param0 plus the other starting points
        param0_points = np.concatenate([[param0], other_param0])

    if not bounded:
        param_bounds = None

    # Create container for results
    out = containers.ChisqPowerSpectrum1D(
        mock=np.arange(nmock, dtype=int),
        param=np.array(signal_model.param_name_fit),
        pol=np.array(pol)[ipol],
        k=nk,
    )

    return (
        signal_model,
        null_model,
        mock_data,
        fit_kwargs,
        param0_points,
        param_bounds,
        fit_cont_for_mocks,
        out,
    )


def powerspectrum1d_min_chisq_fit(
    mcmcfit_cont,
    mcmcfit_cont_for_mocks=None,
    model_kwargs=None,
    param_spec=None,
    param0=None,
    extra_starts=0,
    extra_start_log_bounds=None,
    extra_start_log_samplenegative=True,
    method="L-BFGS-B",
    options=None,
    scale_bound=0.0,
    force_real=True,
    add_mock_to_data=None,
    save_bestfit_models=False,
    use_LOO_covariance=False,
    use_LOO_hartlap=True,
    seed=0,
    n_mc_ks=10000,
    n_mc_ad=10000,
    eps_F=1e-8,
    verbose_notebook=False,
):
    """Compute the minimum chi^2 for 1d power spectrum data and mocks.

    Parameters
    ----------
    mcmcfit_cont : containers.MCMCFitPowerSpectrum1D or str
        Container (or container filename) with information about data,
        mocks, covariance, and signal model.
    mcmcfit_cont_for_mocks : containers.MCMCFitPowerSpectrum1D or str
        Container (or container filename) containing mocks to use for fits.
        If not specified, mocks from `mcmc_fit_cont` are used.
        Default: None.
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
    extra_starts : int, optional
        Also run the optimizer from this number of randomly-chosen points in parameter
        space, and keep the best-fit point out of all the runs. Default: 0.
    extra_start_log_bounds : np.ndarray[nparam, 2], optional
        If specified, `extra_starts` points in parameter space will be randomly chosen
        in log(parameter), with lower and upper log bounds specified by each column of
        this array (e.g. [[-2, 6], [-2, 2]] chooses points with log10(param1) between
        -2 and 6, and log10(param2) between -2 and 2). Default: None.
    extra_start_log_samplenegative : bool, optional
        If sampling starting points in log, randomly choose the sign of each parameter
        for each starting point. This allows for both positive and negative parameter
        values to be sampled, with absolute values with a specified log range.
        Default: True.
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
    use_LOO_covariance : bool, optional
        If True, use leave-one-out covariance for each mock. Default: False.
    use_LOO_hartlap : bool, optional
        Apply Hartlap factor to LOO inverse-covariance. Default: True.
    seed : int, optional
        Random seed for generating starting points for optimizer. Default: 0.
    n_mc_ks : int, optional
        Number of Monte Carlo samples for Monte-Carlo-calibrated KS tests
        that compare set of Delta chi^2 values to specific distributions.
        Default: 10000.
    n_mc_ad : int, optional
        Number of Monte Carlo samples for Monte-Carlo-calibrated AD tests
        that compare set of Delta chi^2 values to specific distributions.
        Default: 10000.
    eps_F : float, optional
        Step size used for numerical approximation of the Jacobian used
        in the L-BFGS-B optimizer when fitting an F distribution.
        Default: 1e-8.
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

    if options is None:
        options = {}

    (
        signal_model,
        null_model,
        mock_data,
        fit_kwargs,
        param0_points,
        param_bounds,
        fit_cont_for_mocks,
        out,
    ) = initialize_powerspectrum1d_min_chisq_ingredients(
        mcmcfit_cont,
        mcmcfit_cont_for_mocks=mcmcfit_cont_for_mocks,
        model_kwargs=model_kwargs,
        param_spec=param_spec,
        param0=param0,
        extra_starts=extra_starts,
        extra_start_log_bounds=extra_start_log_bounds,
        extra_start_log_samplenegative=extra_start_log_samplenegative,
        bounded=method in BOUNDED_MINIMIZATION,
        scale_bound=scale_bound,
        force_real=force_real,
        add_mock_to_data=add_mock_to_data,
        initialize_with_mock=None,
        seed=seed,
    )

    nmock, _, _ = mock_data.shape

    # ---------
    # Compute Delta chi^2 for data
    # ---------

    # Run minimizer for each starting point, saving run that yields lowest
    # chi^2
    for pi, params in enumerate(param0_points):
        test_resd = scipy.optimize.minimize(
            signal_model.negative_log_likelihood,
            params,
            method=method,
            bounds=param_bounds,
            options=options,
        )

        if pi == 0:
            # Store results from first run. If success==False at this step
            # and no other result is better, we'll keep this result.
            bestfit_starting_point_idx = 0
            resd = test_resd
            bestfit_negloglike = signal_model.negative_log_likelihood(resd.x)
        else:
            # If this run succeeds and gets a lower chi^2 than the previous best
            # run, update the best-run variables with this one
            test_negloglike = signal_model.negative_log_likelihood(test_resd.x)
            if test_resd["success"] and (test_negloglike < bestfit_negloglike):
                bestfit_starting_point_idx = pi
                resd = test_resd
                bestfit_negloglike = test_negloglike

    # Save results, along with data itself, to output container
    out.attrs["data_signal_success"] = resd.success
    out.attrs["data_signal_chisq"] = 2.0 * signal_model.negative_log_likelihood(resd.x)
    out.attrs["data_signal_bestfit_param"] = resd.x
    out.attrs["data"] = fit_kwargs["data"][:]
    out.attrs["data_bestfit_starting_point_idx"] = bestfit_starting_point_idx

    # Save chi^2 for null model and data
    out.attrs["data_null_chisq"] = 2.0 * null_model.negative_log_likelihood([])

    # Save best-fit model prediction for data
    if save_bestfit_models:
        # Save best-fit model prediction for data
        out.add_dataset("data_bestfit_model")
        out.datasets["data_bestfit_model"][:] = signal_model.model(
            signal_model.get_all_params(resd.x)
        )

    # ---------
    # Compute Delta chi^2 for mocks
    # ---------

    # Dereference datasets for mock fitting
    success = out["success"][:].view(np.ndarray)
    chisq_null = out["chisq_null"][:].view(np.ndarray)
    chisq_signal = out["chisq_signal"][:].view(np.ndarray)
    bestfit_param = out["bestfit_param"][:].view(np.ndarray)
    mock_bestfit_starting_point_idx = out["mock_bestfit_starting_point_idx"][:].view(
        np.ndarray
    )

    # Initialize dataset to store best-fit model predictions for mocks
    if save_bestfit_models:
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
        if use_LOO_covariance:
            fit_kwargs["inv_cov"], LOO_hartlap_factor = (
                get_powerspectrum1d_LOO_invcovariance(
                    mock_data, mm, fit_cont_for_mocks, hartlap=use_LOO_hartlap
                )
            )
        signal_model.set_data(**fit_kwargs)
        null_model.set_data(**fit_kwargs)

        # Minimize negative log-likelihood for fitting signal model to mock data,
        # and save results
        for pi, params in enumerate(param0_points):
            test_resd = scipy.optimize.minimize(
                signal_model.negative_log_likelihood,
                params,
                method=method,
                bounds=param_bounds,
                options=options,
            )

            if pi == 0:
                # Store results from first run. If success==False at this step
                # and no other result is better, we'll keep this result.
                bestfit_starting_point_idx = 0
                resd = test_resd
                bestfit_negloglike = signal_model.negative_log_likelihood(resd.x)
            else:
                # If this run succeeds and gets a lower chi^2 than the previous best
                # run, update the best-run variables with this one
                test_negloglike = signal_model.negative_log_likelihood(test_resd.x)
                if test_resd["success"] and (test_negloglike < bestfit_negloglike):
                    bestfit_starting_point_idx = pi
                    resd = test_resd
                    bestfit_negloglike = test_negloglike

        success[mm] = resd.success
        chisq_signal[mm] = 2.0 * signal_model.negative_log_likelihood(resd.x)
        bestfit_param[mm] = resd.x
        mock_bestfit_starting_point_idx[mm] = bestfit_starting_point_idx

        if save_bestfit_models:
            out.datasets["mock_bestfit_models"][mm] = signal_model.model(
                signal_model.get_all_params(resd.x)
            )

        # Save chi^2 for null model and mock
        chisq_null[mm] = 2.0 * null_model.negative_log_likelihood([])

    # ---------
    # Compute detection significances based on fitted distributions
    # ---------

    # Fit a chi^2 distribution with a free number of d.o.f. to the
    # Delta chi^2 values for each mock
    dchisq_mocks = chisq_null[success] - chisq_signal[success]
    ndof_mocks_chisq = stats.fit_chi2_to_array(dchisq_mocks)
    out.attrs["chisq_distribution_ndof"] = ndof_mocks_chisq

    # Compute Monte-Carlo-calibrated p-values for KS and AD tests
    # comparing the set of Delta chi^2 values to the best-fit
    # chi^2 distribution
    logger.info("Computing KS p-value for chi^2 distribution")
    ks_p_chisq, _, _ = stats.compute_MC_calibrated_distribution_test(
        dchisq_mocks,
        test="KS",
        dist="chi2",
        seed=seed,
        verbose=False,
        n_mc_sims=n_mc_ks,
    )
    out.attrs["chisq_distribution_ks_pvalue"] = ks_p_chisq
    logger.info("Computing AD p-value for chi^2 distribution")
    ad_p_chisq, _, _ = stats.compute_MC_calibrated_distribution_test(
        dchisq_mocks,
        test="AD",
        dist="chi2",
        seed=seed,
        verbose=False,
        n_mc_sims=n_mc_ad,
    )
    out.attrs["chisq_distribution_ad_pvalue"] = ad_p_chisq

    # Compute p-value and "number of sigmas" for Delta chi^2 value for data,
    # using fitted chi^2 distribution
    dchisq_data = out.attrs["data_null_chisq"] - out.attrs["data_signal_chisq"]
    data_pvalue_chisq = scipy.stats.chi2.sf(dchisq_data, ndof_mocks_chisq)
    data_nsigmas_chisq = scipy.stats.norm.isf(data_pvalue_chisq)
    out.attrs["data_pvalue_chisq"] = data_pvalue_chisq
    out.attrs["data_nsigmas_chisq"] = data_nsigmas_chisq

    # If Hartlap factor was applied to inverse covariance, undo it
    # in Delta chi^2 values
    if use_LOO_covariance and use_LOO_hartlap:
        scaled_dchisq_mocks = dchisq_mocks / LOO_hartlap_factor
        scaled_dchisq_data = dchisq_data / LOO_hartlap_factor
    else:
        scaled_dchisq_mocks = (
            dchisq_mocks / mcmcfit_cont_for_mocks.attrs["hartlap_factor"]
        )
        scaled_dchisq_data = (
            dchisq_data / mcmcfit_cont_for_mocks.attrs["hartlap_factor"]
        )

    # Fit F distribution to Delta chi^2 values
    if use_LOO_covariance:
        n_for_F = nmock - 2
    else:
        n_for_F = nmock - 1
    n_data = len(mcmcfit_cont.attrs["ifit"])
    ndof_mocks_F, _ = stats.fit_F_to_scaled_array(
        scaled_dchisq_mocks, n_for_F, n_data, p_eff_0=None, eps=eps_F
    )
    out.attrs["F_distribution_ndof"] = ndof_mocks_F

    # Compute Monte-Carlo-calibrated p-values for KS and AD tests
    # comparing the set of (appropriately-scaled) Delta chi^2 values
    # to the best-fit F distribution
    logger.info("Computing KS p-value for F distribution")
    ks_p_F, _, _ = stats.compute_MC_calibrated_distribution_test(
        scaled_dchisq_mocks,
        test="KS",
        dist="F",
        n_for_F=n_for_F,
        p_for_F=n_data,
        seed=seed,
        verbose=False,
        n_mc_sims=n_mc_ks,
    )
    out.attrs["F_distribution_ks_pvalue"] = ks_p_F
    logger.info("Computing AD p-value for F distribution")
    ad_p_F, _, _ = stats.compute_MC_calibrated_distribution_test(
        scaled_dchisq_mocks,
        test="AD",
        dist="F",
        n_for_F=n_for_F,
        p_for_F=n_data,
        seed=seed,
        verbose=False,
        n_mc_sims=n_mc_ad,
    )
    out.attrs["F_distribution_ad_pvalue"] = ad_p_F

    # Compute p-value and "number of sigmas" for Delta chi^2 value for data,
    # using fitted F distribution
    scaled_dchisq_data *= stats.F_scaling(n_for_F, n_data, ndof_mocks_F)
    data_pvalue_F = scipy.stats.f.sf(
        scaled_dchisq_data, dfn=ndof_mocks_F, dfd=n_for_F - n_data + 1
    )
    data_nsigmas_F = scipy.stats.norm.isf(data_pvalue_F)
    out.attrs["data_pvalue_F"] = data_pvalue_F
    out.attrs["data_nsigmas_F"] = data_nsigmas_F

    # ---------
    # Compute detection significance based on fitting single amplitude
    # ---------

    # Re-initialize signal model with data
    (
        signal_model,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = initialize_powerspectrum1d_min_chisq_ingredients(
        mcmcfit_cont,
        mcmcfit_cont_for_mocks=mcmcfit_cont_for_mocks,
        model_kwargs=model_kwargs,
        param_spec=param_spec,
        param0=param0,
        extra_starts=extra_starts,
        extra_start_log_bounds=extra_start_log_bounds,
        extra_start_log_samplenegative=extra_start_log_samplenegative,
        bounded=method in BOUNDED_MINIMIZATION,
        scale_bound=scale_bound,
        force_real=force_real,
        add_mock_to_data=add_mock_to_data,
        initialize_with_mock=None,
        seed=seed,
    )

    # Define chi^2 function based on taking best-fit model and
    # scaling with free amplitude (with fiducial value 1.0)
    def signal_chi2_at_amplitude(amp):
        return 2 * signal_model.negative_log_likelihood(
            out.attrs["data_signal_bestfit_param"], amp=amp
        )

    # Find minimum of this chi^2 function
    signal_chi2_min = signal_chi2_at_amplitude(1.0)

    # Find amplitude values on either side of fiducial value
    # where chi^2 function increases by 1
    bf_amp_lo, bf_amp_hi = stats.find_symmetric_roots(
        signal_chi2_at_amplitude, signal_chi2_min + 1.0, 1.0, 0.0, 5.0
    )

    # Detection significance is Delta(amp)/amp.
    # Avearge these values computed with low and high values
    # of amplitude
    data_nsigmas_ampfit_lo = 1.0 / (bf_amp_hi - 1.0)
    data_nsigmas_ampfit_hi = 1.0 / (1.0 - bf_amp_lo)
    data_nsigmas_ampfit = 0.5 * (data_nsigmas_ampfit_lo + data_nsigmas_ampfit_hi)
    out.attrs["data_nsigmas_ampfit"] = data_nsigmas_ampfit

    # Same other useful information
    out.attrs["param0_points"] = param0_points
    out.attrs["seed"] = seed

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
    extra_starts = config.Property(proptype=int)
    extra_start_log_bounds = config.Property(proptype=list)
    extra_start_log_samplenegative = config.Property(proptype=bool)
    method = config.Property(proptype=str)
    options = config.Property(proptype=dict)
    scale_bound = config.Property(proptype=float)
    force_real = config.Property(proptype=bool)
    add_mock_to_data = config.Property(proptype=int)
    save_bestfit_models = config.Property(proptype=bool)
    use_LOO_covariance = config.Property(proptype=bool)
    use_LOO_hartlap = config.Property(proptype=bool)
    n_mc_ks = config.Property(proptype=int)
    n_mc_ad = config.Property(proptype=int)
    eps_F = config.Property(proptype=float)
    seed = config.Property(proptype=int)

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
            elif key in ["mcmcfit_cont", "mcmcfit_cont_for_mocks", "verbose_notebook"]:
                continue
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


class PowerSpectrum1DMinChisqFit_Split(PowerSpectrum1DMinChisqFit):
    """Calls the powerspectrum1d_min_chisq_fit with split-mock approach.

    This task takes two input containers: the first one determines most
    aspects of the procedure, while the second one only provides the mocks
    that will be used for fitting. This allows for the inverse covariance
    to be generated from a different set of mocks than the set used for
    fitting.
    """

    model_kwargs = config.Property(proptype=dict)
    param_spec = config.Property(proptype=dict)
    param0 = config.Property(proptype=list)
    extra_starts = config.Property(proptype=int)
    extra_start_log_bounds = config.Property(proptype=list)
    extra_start_log_samplenegative = config.Property(proptype=bool)
    method = config.Property(proptype=str)
    options = config.Property(proptype=dict)
    scale_bound = config.Property(proptype=float)
    force_real = config.Property(proptype=bool)
    add_mock_to_data = config.Property(proptype=int)
    save_bestfit_models = config.Property(proptype=bool)
    use_LOO_covariance = config.Property(proptype=bool)
    use_LOO_hartlap = config.Property(proptype=bool)
    n_mc_ks = config.Property(proptype=int)
    n_mc_ad = config.Property(proptype=int)
    eps_F = config.Property(proptype=float)
    seed = config.Property(proptype=int)

    def process(self, mcmcfit_cont, mcmcfit_cont_for_mocks):
        """Run the chi^2 minimization.

        Parameters
        ----------
        mcmcfit_cont : containers.MCMCFitPowerSpectrum1D
            Container with information about data, mocks used for
            covariance computation, covariance, and signal model.
        mcmcfit_cont_for_mocks : containers.MCMCFitPowerSpectrum1D
            Container with mocked to be used for fitting.

        Returns
        -------
        out : containers.ChisqPowerSpectrum1D
            Container containing chi^2 results and associated information.
        """
        out = powerspectrum1d_min_chisq_fit(
            mcmcfit_cont,
            mcmcfit_cont_for_mocks=mcmcfit_cont_for_mocks,
            verbose_notebook=False,
            **self.kwargs,
        )

        return out


def combine_dchi2_results(cont_list):
    """Make array of highest :math:`\Delta\chi^2` value for each noise mock.

    Parameters
    ----------
    cont_list : list
        List of `fitstack.containers.ChisqPowerSpectrum1D` containers with
        the results of :math:`\chi^2` minimization.

    Returns
    -------
    dchi2_max_arr : np.ndarray[nmocks]
        Array of highest :math:`\Delta\chi^2` value for each mock,
        over :math:`\Delta\chi^2`values computed from each input
        container.
    """

    _DCHI2_FAILURE_VALUE = 0

    # Make array of dchi2 values for each key
    dchi2_2darr = []
    for cont in cont_list:
        # Get dchi2 values for this container
        dchi2_single = np.abs(cont["chisq_null"][:] - cont["chisq_signal"][:])
        # For any mock where the minimizer reported failure,
        # replace its dchi2 value with a small value, so that
        # it's ignored when we take the maximum dchi2 over
        # all keys
        dchi2_single[~cont["success"][:]] = _DCHI2_FAILURE_VALUE

        dchi2_2darr.append(dchi2_single)

    dchi2_2darr = np.array(dchi2_2darr)

    # For each mock, take maximum dchi2 value over all containers
    dchi2_max_arr = np.max(dchi2_2darr, axis=0)

    # If there's a mock for which no key had a successful fit,
    # remove it from the list by checking its dchi2 value
    dchi2_max_arr = dchi2_max_arr[dchi2_max_arr > _DCHI2_FAILURE_VALUE]

    return dchi2_max_arr
