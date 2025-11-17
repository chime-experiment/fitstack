"""Perform an MCMC fit of a model to a source stack."""

import logging
import inspect

import numpy as np
import emcee
import h5py

from caput import config, pipeline

from draco.util import tools
from draco.core import task
from draco.core.io import _list_or_glob

from . import containers
from . import utils
from . import models

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)]
    )


SIMULATION_MODELS = [
    c.__name__
    for c in [
        models.SimulationTemplate,
        models.AutoSimulationTemplate1D,
        models.AutoSimulationTemplate2Dto1D,
    ]
    + list(_all_subclasses(models.SimulationTemplate))
    + list(_all_subclasses(models.AutoSimulationTemplate1D))
    + list(_all_subclasses(models.AutoSimulationTemplate2Dto1D))
]

PS1D_SIMULATION_MODELS = [
    c.__name__
    for c in [models.AutoSimulationTemplate1D]
    + list(_all_subclasses(models.AutoSimulationTemplate1D))
]

PS2D_SIMULATION_MODELS = [
    c.__name__
    for c in [models.AutoSimulationTemplate2Dto1D]
    + list(_all_subclasses(models.AutoSimulationTemplate2Dto1D))
]

PERCENTILE = [2.5, 16, 50, 84, 97.5]

# Co-pols are stored as XX, YY for stacking and XX-XX, YY-YY
# for power spectra. Internally, we always use XX, YY, but we
# need to map these to the relevant strings depending on the input
# data.
_STACKING_POLNAME = {"XX": "XX", "YY": "YY", "I": "I"}
_PS_POLNAME = {"XX": "XX-XX", "YY": "YY-YY", "I": "I-I", "Q": "Q-Q"}


def initialize_mcmc_ingredients(
    data,
    mocks,
    data_2d=None,
    transfer=None,
    template=None,
    pol_fit="joint",
    required_pol=None,
    pol_stokes=False,
    model_name="Exponential",
    scale=1e6,
    nwalker=32,
    nsample=15000,
    max_freq=None,
    min_k=None,
    max_k=None,
    flag_before=False,
    normalize_template=False,
    mean_subtract=True,
    recompute_weight=True,
    model_kwargs=None,
    param_spec=None,
    seed=None,
    flag_ind=None,
    hartlap=True,
    force_real=True,
):
    """Initialize ingredients for MCMC.

    See run_mcmc docstring for parameter info.

    Returns
    -------
    results : containers.MCMCFit
        Container data products and ancillary data products, as well
        as the covariance and precision matrix inferred from the mocks.
    model : models.Model
        Model to be used in chain.
    eval_kwargs : dict
        Keyword arguments to be used when evaluating model.
    npol : int
        Number of polarizations to fit.
    nx : int
        Number of data points to fit.
    """

    # Function to take real part if force_real is set
    def _re(x):
        return np.real(x) if force_real else x

    if isinstance(data, str):
        with h5py.File(utils.find_file(data), "r") as handler:
            container_type = handler.attrs["__memh5_subclass"]
            if container_type == "draco.core.containers.FrequencyStackByPol":
                polname = _STACKING_POLNAME
            elif container_type in [
                "draco.core.containers.PowerSpectrum2D",
                "draco.core.containers.PowerSpectrum1D",
            ]:
                polname = _PS_POLNAME
            else:
                raise RuntimeError(
                    f"Container type of data argument ({container_type})"
                    " not recognized"
                )
    else:
        if isinstance(data, containers.FrequencyStackByPol):
            polname = _STACKING_POLNAME
        else:
            polname = _PS_POLNAME

    if pol_stokes:
        if required_pol is None:
            required_pol = [polname["I"], polname["Q"]]
        combine_pol = False
    else:
        if required_pol is None:
            required_pol = [polname["XX"], polname["YY"]]
        combine_pol = (
            True
            if ("DualPol" not in model_name) and ("Split" not in model_name)
            else False
        )

    if param_spec is None:
        param_spec = {}

    if model_kwargs is None:
        model_kwargs = {}

    # Load the data
    if isinstance(data, str):
        data = utils.load_pol(utils.find_file(data), pol=required_pol)

    # From the input container, determine the dataset corresponding to the observable
    # (stack or power spectrum)
    if isinstance(data, containers.FrequencyStackByPol):
        dset = "stack"
    elif isinstance(data, containers.PowerSpectrum1D):
        dset = "spectrum"
    else:
        raise RuntimeError(f"Input container {type(data)} not supported")

    # If using 1d power spectrum measurements with a 2d power spectrum model, load
    # 2d measurements
    ps2d_model = model_name in PS2D_SIMULATION_MODELS
    if dset == "spectrum" and ps2d_model:
        if data_2d is None:
            raise RuntimeError(
                "Must specify data_2d if fitting 1d power spectrum "
                "measurements with model that starts in 2d"
            )
        if isinstance(data_2d, str):
            data_2d = utils.load_pol(utils.find_file(data_2d), pol=required_pol)

    # Load the transfer function (not implemented for power spectra)
    if transfer is not None and isinstance(transfer, str):
        if dset == "spectrum":
            raise NotImplementedError(
                "Transfer function convolution not implemented for power spectrum"
            )
        transfer = utils.load_pol(utils.find_file(transfer), pol=required_pol)

    # Load the templates
    if template is not None:
        template = utils.load_mocks(template, pol=required_pol)

    # Load the mock catalogs or noise power spectra
    mocks = utils.load_mocks(mocks, pol=required_pol)
    nmock = mocks.index_map["mock"].size

    # If requested, use the inverse variance over the mock catalogs
    # or noise power spectra as the weight when averaging over polarisations.
    if recompute_weight:

        axes = mocks[dset].attrs["axis"][1:]

        for container in [data, mocks, template, transfer]:
            if container is not None:
                if dset == "stack":
                    inv_var = tools.invert_no_zero(np.var(_re(mocks.stack[:]), axis=0))
                    expand = tuple(
                        slice(None) if ax in axes else None
                        for ax in container.weight.attrs["axis"]
                    )
                    container.weight[:] = inv_var[expand]
                else:
                    variance = np.var(_re(mocks.spectrum[:]), axis=0)
                    expand = tuple(
                        slice(None) if ax in axes else None
                        for ax in container.var.attrs["axis"]
                    )
                    container.var[:] = variance[expand]

    # For the simulation template, we need to provide parameters to average the polarisations
    if model_name in SIMULATION_MODELS:
        if dset == "stack":
            if "Split" not in model_name:
                model_kwargs["weight"] = (
                    inv_var if recompute_weight else _re(data.weight[:])
                )
        else:
            if ps2d_model:
                # If our power spectrum model starts from 2d, take weights and signal_mask
                # from data_2d
                _, weight_meas_2d, _, _, signal_mask_2d, _ = utils.initialize_pol(
                    data_2d,
                    pol=required_pol,
                    combine=combine_pol,
                    return_signal_mask_and_neff=True,
                )
                model_kwargs["weight"] = _re(weight_meas_2d)
                model_kwargs["signal_mask"] = signal_mask_2d

                model_kwargs["nbins"] = data.k1D.shape[-1]
            else:
                model_kwargs["weight"] = tools.invert_no_zero(data.var[:])

        model_kwargs["pol"] = required_pol
        model_kwargs["combine"] = combine_pol
        model_kwargs["sort"] = True

    # Initialize data arrays.
    # We use x to stand for either frequencies or k values, depending on input data type
    data_meas, weight_meas, pol, x_meas = utils.initialize_pol(
        data, pol=required_pol, combine=combine_pol
    )

    # Sort frequencies or k values, then determine which values to use in fit
    isort = np.argsort(_re(x_meas), axis=-1)
    x = np.take_along_axis(_re(x_meas), isort, axis=-1)
    nx = x.shape[-1]

    if dset == "stack":
        if max_freq is None:
            max_freq = 1e10
        x_flag = np.abs(x) < max_freq
    else:
        if min_k is None:
            min_k = 0.0
        if max_k is None:
            max_k = 1e10
        x_flag = (np.abs(x) > min_k) & (np.abs(x) < max_k)

    # Use pol-independent flags, for simplicity
    x_1d_flag = np.all(x_flag, axis=0)
    x_1d_index = np.flatnonzero(x_1d_flag)
    x_1d_slice = slice(x_1d_index[0], x_1d_index[-1] + 1)

    # If working with stacks, define 1d frequency axis for later convenience
    if dset == "stack":
        freq = np.mean(x, axis=0)

    # Sort data and weight arrays
    data_meas = scale * np.take_along_axis(_re(data_meas), isort, axis=-1)
    weight_meas = np.take_along_axis(_re(weight_meas), isort, axis=-1) / scale**2
    npol = len(pol)

    # Initialize array for mocks
    mock_meas, _, _, _ = utils.initialize_pol(
        mocks, pol=required_pol, combine=combine_pol
    )
    mock_meas = scale * np.take_along_axis(
        _re(mock_meas), isort[np.newaxis, ...], axis=-1
    )

    # Initialize array for transfer function
    if transfer is not None:
        transfer_meas, _, _, _ = utils.initialize_pol(
            transfer, pol=required_pol, combine=combine_pol
        )
        transfer_meas = np.take_along_axis(_re(transfer_meas), isort, axis=-1)

    # Initialize array for template
    if template is not None:
        # Use the mean value of the template over realizations
        template = utils.average_data(
            template, pol=required_pol, combine=combine_pol, sort=True
        )
        template_meas = scale * _re(template[dset][:])

        if normalize_template:
            max_template = np.max(
                template_meas[..., x_1d_slice], axis=-1, keepdims=True
            )
            template_meas = template_meas / max_template

    # Subtract mean value of mocks from the mocks and data
    if mean_subtract:
        logger.info("Subtracting the mean value of the mocks.")
        mu = np.mean(mock_meas, axis=0)
        mock_meas = mock_meas - mu[np.newaxis, ...]
        data_meas = data_meas - mu

    # Prepare the model
    Model = getattr(models, model_name)
    model = Model(seed=seed, force_real=force_real, **{**model_kwargs, **param_spec})

    # Calculate the covariance over mocks
    cov_flat = utils.covariance(mock_meas.reshape(nmock, -1), corr=False)
    cov = utils.unravel_covariance(cov_flat, npol, nx)

    # Determine the polarisation to fit
    if pol_fit in polname.keys():
        ipol = pol.index(polname[pol_fit])
        npol_fit = 1

        C = cov[ipol, ipol]
        ifit = x_1d_index

    elif pol_fit == "joint":
        if pol_stokes:
            ipol = np.array([pol.index(pstr) for pstr in [polname["I"], polname["Q"]]])
        else:
            ipol = np.array(
                [pol.index(pstr) for pstr in [polname["XX"], polname["YY"]]]
            )
        npol_fit = len(ipol)

        C = utils.ravel_covariance(cov[ipol][:, ipol])
        ifit = np.concatenate(tuple([p * nx + x_1d_index for p in range(npol_fit)]))

    else:
        raise ValueError(
            f"Do not recognize polarisation {pol_fit} "
            "(possible values are 'XX', 'YY', 'I', 'Q', or 'joint')"
        )

    # Apply extra flagging if specified
    if flag_ind is not None:
        logger.debug(f"Flagging out extra data: starting with {len(ifit)} samples.")
        ifit = [fi for ii, fi in enumerate(ifit) if ii not in flag_ind]
        logger.debug(f"Ending with {len(ifit)} samples.")
    else:
        logger.debug("No samples flagged out.")

    # Make array of coordinates for MCMC, as list of (pol, coord_value) tuples
    pol = np.array(pol)
    x_coord_name = "freq" if dset == "stack" else "k"
    x_for_mcmc = np.zeros(
        nx * npol_fit, dtype=[("pol", "U8"), (x_coord_name, np.float64)]
    )
    for p, pstr in enumerate(np.atleast_1d(pol[ipol])):
        slc = slice(p * nx, (p + 1) * nx)
        x_for_mcmc["pol"][slc] = pstr
        x_for_mcmc[x_coord_name][slc] = x[p]

    # Create results container
    if dset == "stack":
        results = containers.MCMCFit1D(
            x=x_for_mcmc,
            freq=freq,
            pol=pol,
            mock=nmock,
            walker=nwalker,
            step=nsample,
            param=np.array(model.param_name),
            percentile=np.array(PERCENTILE),
        )
        results["weight"][:] = weight_meas
        results["freq_flag"][:] = x_1d_flag

    else:
        results = containers.MCMCFitPowerSpectrum1D(
            x=x_for_mcmc,
            k=nx,
            pol=pol,
            mock=nmock,
            walker=nwalker,
            step=nsample,
            param=np.array(model.param_name),
            percentile=np.array(PERCENTILE),
            cosmology=data.cosmology,
        )
        results["var"][:] = tools.invert_no_zero(weight_meas)
        results["k_flag"][:] = x_1d_flag
        results["k1D"][:] = x

    results.attrs["seed"] = str(model.seed)
    results.attrs["model"] = model_name
    results.attrs["pol_fit"] = (
        polname[pol_fit] if pol_fit in polname.keys() else pol_fit
    )

    results["mock"][:] = mock_meas
    results[dset][:] = data_meas

    results["fixed"][:] = True
    results["fixed"][:][model.fit_index] = False

    if transfer is not None:
        results.add_dataset("transfer_function")
        results["transfer_function"][:] = transfer_meas

    if template is not None:
        results.add_dataset("template")
        results["template"][:] = template_meas

    results["cov"][:] = cov
    results["error"][:] = np.sqrt(np.diag(cov_flat).reshape(npol, nx))

    # Invert the covariance matrix to obtain the precision matrix
    Cinv = np.zeros_like(C)

    if flag_before:
        C = C[ifit][:, ifit]

    Cinvfit = np.linalg.pinv(C)

    # Compute and apply the Hartlap factor to the inverse covariance
    if hartlap:
        hartlap_factor = (nmock - Cinvfit.shape[0] - 2.0) / (nmock - 1.0)
        Cinvfit *= hartlap_factor
    else:
        hartlap_factor = 1.0

    if not flag_before:
        Cinvfit = Cinvfit[ifit][:, ifit]

    for ii, oi in enumerate(ifit):
        Cinv[oi, ifit] = Cinvfit[ii, :]

    results["precision"][:] = Cinv
    results["flag"][:] = np.diag(Cinv) > 0.0

    # Set the data for this polarisation
    y = data_meas[ipol]

    if dset == "stack":
        # freq is 1d array of frequencies
        fit_kwargs = {"freq": freq, "data": y, "inv_cov": Cinv}
        eval_kwargs = {"freq": freq}
    else:
        # Reminder: x is 2d array of [pol,k]
        fit_kwargs = {"k1D": x[ipol], "data": y, "inv_cov": Cinv}
        eval_kwargs = {"k1D": x}

    if transfer is not None:
        fit_kwargs["transfer"] = transfer_meas[ipol]
        eval_kwargs["transfer"] = transfer_meas[:]
    else:
        fit_kwargs["transfer"] = None
        eval_kwargs["transfer"] = None

    if template is not None:
        fit_kwargs["template"] = template_meas[ipol]
        eval_kwargs["template"] = template_meas[:]

    if (
        model_name in SIMULATION_MODELS
        and ("DualPol" not in model_name)
        and ("Split" not in model_name)
    ):
        fit_kwargs["pol_sel"] = ipol
        eval_kwargs["pol_sel"] = slice(None)

    # Save model_kwargs, param_spec, and a few other quantities
    # to results container for later reference
    results.attrs["model_kwargs"] = model_kwargs
    results.attrs["param_spec"] = param_spec
    results.attrs["pol_sel"] = ipol
    results.attrs["ifit"] = ifit
    results.attrs["flag_before"] = flag_before
    results.attrs["hartlap_factor"] = hartlap_factor

    model.set_data(**fit_kwargs)

    return results, model, eval_kwargs, npol, nx


def run_mcmc(
    data,
    mocks,
    data_2d=None,
    transfer=None,
    template=None,
    pol_fit="joint",
    required_pol=None,
    pol_stokes=False,
    model_name="Exponential",
    scale=1e6,
    nwalker=32,
    nsample=15000,
    max_freq=None,
    min_k=None,
    max_k=None,
    flag_before=False,
    normalize_template=False,
    mean_subtract=True,
    recompute_weight=True,
    model_kwargs=None,
    param_spec=None,
    seed=None,
    flag_ind=None,
    hartlap=True,
    force_real=True,
    progress=True,
):
    """Fit a model to the source stack or power spectrum using an MCMC.

    Parameters
    ----------
    data : FrequencyStackByPol, PowerSpectrum1D, or str
        Measurements of stacking or power spectrum.
        This can either be a FrequencyStackByPol or PowerSpectrum1D
        container, or the name of a file that holds such a container
        and will be loaded from disk.
    mocks : container, list of containers, str, or list of str
        Mocks for estimating a noise covariance.
        This can either be a MockFrequencyStackByPol or
        MockPowerSpectrum1D container, a list of FrequencyStackByPol or
        PowerSpectrum1D containers, or the name of a file or a list of
        filenames that hold such containers and will be loaded from disk.
    data_2d : PowerSpectrum2D or str
        Measurements of 2d power spectrum, either as a PowerSpectrum2D
        container or filename. When fitting to 1d power spectrum
        measurements with a model that starts in 2d, weights and
        (kpara,kperp) masking will be taken from here. Ignored if not
        needed.
    transfer : FrequencyStackByPol or str
        The transfer function of the pipeline (only implemented for stacking).
        The model for the stacked
        signal will be convolved with the transfer function prior
        to comparing to the data.  This can either be a
        FrequencyStackByPol container or the name of a file that
        holds such a container and will be loaded from disk.
        If None, then a transfer function is not applied.  Default is None.
    template : FrequencyStackByPol, PowerSpectrum1D, or str
        Template for the stacked signal.  This can either be a
        FrequencyStackByPol or PowerSpectrum1D container, or the name of
        a file that holds such a container and will be loaded from disk.
        Note that not all models require templates.  Default is None.
    pol_fit : {"XX"|"YY"|"I"|"Q"|"joint"}
        Polarisation to fit.  Here "I" refers to the weighted sum of the
        "XX" and "YY" polarisations for stacking measurements and the
        unweighted sum of "XX" and "YY" for power spectrum measurements.
        "joint" refers to a simultaneous fit to the "XX" and "YY" or "I"
        and "Q" polarisations.
    required_pol : list
        Polarizations to load from file.  If None, defaults to ["XX", "YY"]
        for stacking measurements and ["I-I", "Q-Q] for power spectrum
        measuerements.
    pol_stokes : bool
        If True, assume that all input files contain Stokes parameters
        instead of instrumental polarisations. Default: False.
    model_name : {"DeltaFunction"|"Exponential"|"ScaledShiftedTemplate"|
                  "SimulationTemplate"|"SimulationTemplateFoG"|
                  "SimulationTemplateFoGAltParam"|"AutoConstant"|
                  "AutoSimulationTemplate2Dto1D"}
        Name of the model to fit.  Specify the class name from the
        fitstack.models module.
    scale : float
        All data will be scaled by this quantity.  Default is 1e6, under the
        assumption that the signal in a stacking analysis is in units of Jy / beam
        and we would like to convert to micro-Jy / beam.
        Set to 1.0 if you do not want to scale the data.
    nwalker : int
        Number of walkers to use in the MCMC fit.  Default is 32.
    nsample : int
        Number of steps that each walker will take.  Default is 15000.
    max_freq : float
        The maximum frequency offset to include in a stacking fit.  If this None,
        then all frequency offsets that are present in the data stack
        will be included.  Default is None.
    min_k, max_k : float
        The minimum or maximum k values to include in a power spectrum fit.
        Default is None.
    flag_before : bool
        Only relevant if max_freq, min_k, or max_k is not None.
        The frequency offset flag will be applied prior to calculating the
        inverse of the covariance matrix. Default is False.
    normalize_template : bool
        Divide the template by its maximum value prior to fitting.
        Default is False.
    mean_subtract : bool
        Subtract the sample mean of the mocks from the data prior to fitting.
        Default is True.
    recompute_weight : bool
        Set the weight dataset to the inverse variance over the mock catalogs
        in a stacking analysis, or set the 1D `var` dataset to the variance over
        the noise power spectra in a power spectrum analysis.
        This is only used when averaging the "XX" and "YY" polarisations to
        determine the "I" polarisation.  Otherwise whatever weight/var dataset
        is saved to the file will be used.  Default is True.
    param_spec : dict
        Dictionary that specifies the prior distribution for each parameter.
        See the docstring for the models.Model attribute for the correct format.
    model_kwargs : dict
        Dictionary that contains any keyword arguments that should be passed
        to the model class at initialization.
    seed : int
        Seed to use for random number generation.  If the seed is not provided,
        then a random seed will be taken from system entropy.
    flag_ind : list
        List of extra indices to flag. These are indices into the flattened data *after*
        all other selections have been applied.
    hartlap : bool
        Apply the Hartlap factor to the inverse covariance computed from mocks.
        Default: True.
    force_real : bool
        Force input datasets to be real. Assumes that input datasets have
        been previously examined to verify that imaginary parts are small and/or
        unimportant. Default: True.
    progress : bool
        Whether to display emcee progress bar. Default: True.

    Returns
    -------
    results : MCMCFit
        Container with the results of the fit, including
        parameter chains, chi-squared chains, autocorrelation length,
        acceptance franction, parameter percentiles, best-fit model,
        and model percentiles.  Also includes all data products
        and ancillary data products, as well as the covariance and
        precision matrix inferred from the mocks.
    """

    # Initialize MCMCFit container for results, theory model, kwargs for
    # model, and lengths of data to fit
    results, model, eval_kwargs, npol, nx = initialize_mcmc_ingredients(
        data,
        mocks,
        data_2d=data_2d,
        transfer=transfer,
        template=template,
        pol_fit=pol_fit,
        pol_stokes=pol_stokes,
        model_name=model_name,
        scale=scale,
        nwalker=nwalker,
        nsample=nsample,
        max_freq=max_freq,
        min_k=min_k,
        max_k=max_k,
        flag_before=flag_before,
        normalize_template=normalize_template,
        mean_subtract=mean_subtract,
        recompute_weight=recompute_weight,
        model_kwargs=model_kwargs,
        param_spec=param_spec,
        seed=seed,
        flag_ind=flag_ind,
        hartlap=hartlap,
        force_real=force_real,
    )

    # Set seed for emcee
    if seed is not None:
        np.random.seed(seed)

    # Determine starting point for chains in parameter space
    pos = np.array([model.draw_random_parameters() for ww in range(nwalker)])

    nwalker, ndim = pos.shape

    nparam = len(model.param_name)

    # Create the sampler and run the MCMC
    sampler = emcee.EnsembleSampler(nwalker, ndim, model.log_probability_sampler)

    sampler.run_mcmc(model.forward_transform_sampler(pos), nsample, progress=progress)

    chain = model.backward_transform_sampler(sampler.get_chain())

    # Compute the chisq
    chisq = results["chisq"][:].view(np.ndarray)
    for ss in range(nsample):
        for ww in range(nwalker):
            chisq[ss, ww] = -2.0 * model.log_likelihood(chain[ss, ww])

    # Find the minimum chisq
    chain_all = (
        np.ones((nsample, nwalker, nparam), dtype=chain.dtype) * model.default_values
    )
    chain_all[:, :, model.fit_index] = chain

    imin = np.unravel_index(np.argmin(chisq), (nsample, nwalker))

    theta_min = chain_all[imin]

    # Save the results to the output container
    results["chain"][:] = chain_all

    results["autocorr_time"][:] = 0.0
    results["autocorr_time"][:][model.fit_index] = sampler.get_autocorr_time(quiet=True)

    results["acceptance_fraction"][:] = sampler.acceptance_fraction

    results["model_min_chisq"][:] = model.model(theta_min, **eval_kwargs)

    # Discard burn in and thin the chains
    flat_samples = results.samples(flat=True)
    if len(flat_samples) == 0:
        raise RuntimeError(
            "After thinning and burn-in removal, chain has no samples remaining!\n"
            "Re-run with larger number of samples."
        )

    # Compute percentiles of the posterior distribution
    q = np.percentile(flat_samples, PERCENTILE, axis=0).T
    dq = np.diff(q, axis=1)

    results["percentile"][:] = q
    results["median"][:] = q[:, 2]
    results["span_lower"][:] = dq[:, 1]
    results["span_upper"][:] = dq[:, 2]

    # Compute percentiles of the model
    mdl = np.zeros((flat_samples.shape[0], npol, nx), dtype=np.float32)
    for ss, theta in enumerate(flat_samples):
        mdl[ss] = model.model(theta, **eval_kwargs)

    results["model_percentile"][:] = np.percentile(mdl, PERCENTILE, axis=0).transpose(
        1, 2, 0
    )

    # Return results container
    return results


class RunMCMC(task.SingleTask):
    """Pipeline task that calls the run_mcmc function.

    Enables the user to call the run_mcmc method with caput-pipeline,
    which provides many useful features including profiling, job script
    generation, job templating, and saving the results to disk.

    Attributes
    ----------
    max_iter : int
        Number of times to call the run_mcmc method.
        Defaults to 1.

    See the arguments of the run_mcmc method for a list of
    additional attributes and their default values.
    """

    max_iter = config.Property(proptype=int, default=1)
    data_2d = config.Property(proptype=str, default=None)

    data = config.Property(proptype=str)
    mocks = config.Property(proptype=_list_or_glob)
    transfer = config.Property(proptype=str)
    template = config.Property(proptype=_list_or_glob)

    pol_fit = config.Property(proptype=str)
    required_pol = config.Property(proptype=list)
    pol_stokes = config.Property(proptype=bool)
    model_name = config.Property(proptype=str)
    scale = config.Property(proptype=float)

    nwalker = config.Property(proptype=int)
    nsample = config.Property(proptype=int)

    max_freq = config.Property(proptype=float)
    min_k = config.Property(proptype=float)
    max_k = config.Property(proptype=float)
    flag_before = config.Property(proptype=bool)
    normalize_template = config.Property(proptype=bool)
    mean_subtract = config.Property(proptype=bool)
    recompute_weight = config.Property(proptype=bool)

    param_spec = config.Property(proptype=dict)
    model_kwargs = config.Property(proptype=dict)
    seed = config.Property(proptype=int)
    flag_ind = config.list_type(type_=int)
    hartlap = config.Property(proptype=bool)

    force_real = config.Property(proptype=bool)
    progress = config.Property(proptype=bool)

    def setup(self):
        """Prepare all arguments to the run_mcmc function."""

        # Use the default values from the run_mcmc method,
        # so we do not have to repeat them in two places.
        signature = inspect.signature(run_mcmc)
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
                    "RunMCMC does not have a property corresponding "
                    f"to the {key} keyword argument to run_mcmc."
                )

    def process(self):
        """Fit a model to the source stack using an MCMC."""

        if self._count == self.max_iter:
            raise pipeline.PipelineStopIteration

        result = run_mcmc(**self.kwargs)

        return result
