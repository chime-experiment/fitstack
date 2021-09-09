"""Perform an MCMC fit of a model to a source stack."""
import logging
import inspect

import numpy as np
import emcee

from caput import config

from draco.util import tools
from draco.core import task
from draco.core.io import _list_or_glob

from . import containers
from . import utils
from . import models

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PERCENTILE = [2.5, 16, 50, 84, 97.5]


# main script
def run_mcmc(
    data,
    mocks,
    transfer=None,
    template=None,
    pol_fit="joint",
    model_name="Exponential",
    scale=1e6,
    nwalker=32,
    nsample=15000,
    max_freq=None,
    flag_before=True,
    normalize_template=False,
    mean_subtract=True,
    recompute_weight=True,
    prior_spec=None,
    seed=None,
):
    """Fit a model to the source stack using an MCMC.

    Parameters
    ----------
    data : FrequencyStackByPol or str
        Stack of the data on the sources.  This can either be a
        FrequencyStackByPol container or the name of a file that
        holds such a container and will be loaded from disk.
    mocks : MockFrequencyStackByPol, list of FrequencyStackByPol, str, list of str
        Stack of the data on a set of random mock catalogs.
        This can either be a MockFrequencyStackByPol container
        or a list of FrequencyStackByPol containers.  It can also
        be the name of a file or a list of filenames that hold
        such containers and will be loaded from disk.
    transfer : FrequencyStackByPol or str
        The transfer function of the pipeline.  The model for the stacked
        signal will be convolved with the transfer function prior
        to comparing to the data.  This can either be a
        FrequencyStackByPol container or the name of a file that
        holds such a container and will be loaded from disk.
        If None, then a transfer function is not applied.  Default is None.
    template : FrequencyStackByPol or str
        Template for the stacked signal.  This can either be a
        FrequencyStackByPol container or the name of a file that
        holds such a container and will be loaded from disk.
        Note that not all models require templates.  Default is None.
    pol_fit : {"XX"|"YY"|"I"|"joint"}
        Polarisation to fit.  Here "I" refers to the weighted sum of the
         "XX" and "YY" polarisations and "joint" refers to a simultaneous
        fit to the "XX" and "YY" polarisations.
    model_name : {"DeltaFunction"|"Exponential"|"ScaledShiftedTemplate"}
        Name of the model to fit.  Specify the class name from the
        fitstack.models module.
    scale : float
        All data will be scaled by this quantity.  Default is 1e6, under the
        assumption that the stacked signal is in units of Jy / beam and we would
        like to convert to micro-Jy / beam.  Set to 1.0 if you do not want to
        scale the data.
    nwalker : int
        Number of walkers to use in the MCMC fit.  Default is 32.
    nsample : int
        Number of steps that each walker will take.  Default is 15000.
    max_freq : float
        The maximum frequency offset to include in the fit.  If this None,
        then all frequency offsets that are present in the data stack
        will be included.  Default is None.
    flag_before : bool
        Only relevant if max_freq is not None.  The frequency offset flag will
        be applied prior to calculating the inverse of the covariance matrix.
        Default is True.
    normalize_template : bool
        Divide the template by it's maximum value prior to fitting.
        Default is False.
    mean_subtract : bool
        Subtract the sample mean of the mocks from the data prior to fitting.
        Default is True.
    recompute_weight : bool
        Set the weight dataset to the inverse variance over the mock catalogs.
        This is only used when averaging the "XX" and "YY" polarisations to
        determine the "I" polarisation.  Otherwise whatever weight dataset
        is saved to the file will be used.  Default is True.
    prior_spec : dict
        Dictionary that specifies the prior distribution for each parameter.
        See the docstring for the models.Model attribute for the correct format.
    seed : int
        Seed to use for random number generation.  If the seed is not provided,
        then a random seed will be taken from system entropy.

    Returns
    -------
    results : MockStack1D
        Container with the results of the fit, including
        parameter chains, chi-squared chains, autocorrelation length,
        acceptance franction, parameter percentiles, best-fit model,
        and model percentiles.  Also includes all data products
        and ancillary data products, as well as the covariance and
        precision matrix inferred from the mocks.
    """

    required_pol = ["XX", "YY"]
    combine_pol = True

    # Load the data
    if isinstance(data, str):
        data = utils.find_file(data)
        pol_sel = utils.determine_pol_sel(data[0], pol=required_pol)
        data = containers.FrequencyStackByPol.from_file(data[0], pol_sel=pol_sel)

    # Load the transfer function
    if transfer is not None and isinstance(transfer, str):
        transfer = utils.find_file(transfer)
        pol_sel = utils.determine_pol_sel(transfer, pol=required_pol)
        transfer = containers.FrequencyStackByPol.from_file(transfer, pol_sel=pol_sel)

    # Load the templates
    if template is not None:
        template = utils.load_mocks(template, pol=required_pol)

    # Load the mock catalogs
    mocks = utils.load_mocks(mocks, pol=required_pol)
    nmock = mocks.index_map["mock"].size

    # If requested, use the inverse variance over the mock catalogs as the weight
    # when averaging over polarisations.
    if recompute_weight:

        inv_var = tools.invert_no_zero(np.var(mocks.stack[:], axis=0))
        axes = mocks.stack.attrs["axis"][1:]

        for container in [data, mocks, template, transfer]:
            if container is not None:
                expand = tuple(
                    slice(None) if ax in axes else None
                    for ax in container.weight.attrs["axis"]
                )
                container.weight[:] = inv_var[expand]

    # Determine the frequencies to fit
    freq = data.freq[:]
    nfreq = freq.size

    isort = np.argsort(freq)
    freq = freq[isort]

    if max_freq is not None:
        freq_flag = np.abs(freq) < max_freq
    else:
        freq_flag = np.ones(nfreq, dtype=bool)

    freq_index = np.flatnonzero(freq_flag)
    freq_slice = slice(freq_index[0], freq_index[-1] + 1)

    # Initialize all arrays
    data_stack, weight_stack, pol = utils.initialize_pol(
        data, pol=required_pol, combine=combine_pol
    )
    data_stack = scale * data_stack[..., isort]
    weight_stack = weight_stack[..., isort] / scale ** 2
    npol = len(pol)

    mock_stack, _, _ = utils.initialize_pol(
        mocks, pol=required_pol, combine=combine_pol
    )
    mock_stack = scale * mock_stack[..., isort]

    if transfer is not None:
        transfer_stack, _, _ = utils.initialize_pol(
            transfer, pol=required_pol, combine=combine_pol
        )
        transfer_stack = transfer_stack[..., isort]

    if template is not None:
        template_stack, _, _ = utils.initialize_pol(
            template, pol=required_pol, combine=combine_pol
        )

        # Use the mean value of the template over realizations
        template_stack = scale * np.mean(template_stack, axis=0)[..., isort]

        if normalize_template:
            max_template = np.max(
                template_stack[..., freq_slice], axis=-1, keepdims=True
            )
            template_stack = template_stack / max_template

    # Subtract mean value of mocks
    if mean_subtract:
        logger.info("Subtracting the mean value of the mocks.")
        mu = np.mean(mock_stack, axis=0)
        mock_stack = mock_stack - mu[np.newaxis, ...]
        data_stack = data_stack - mu

    # Prepare the model
    Model = getattr(models, model_name)
    model = Model(seed=seed, **prior_spec)

    param_name = model.param_name
    nparam = len(param_name)

    # Calculate the covariance over mocks
    cov_flat = utils.covariance(mock_stack.reshape(nmock, -1), corr=False)

    cov = utils.unravel_covariance(cov_flat, npol, nfreq)

    # Determine the polarisation to fit
    if pol_fit in ["XX", "YY", "I"]:
        ipol = pol.index(pol_fit)
        npol_fit = 1

        C = cov[ipol, ipol]
        ifit = freq_slice

    elif pol_fit == "joint":
        ipol = np.array([pol.index(pstr) for pstr in ["XX", "YY"]])
        npol_fit = len(ipol)

        C = utils.ravel_covariance(cov[ipol][:, ipol])
        ifit = np.concatenate(tuple([p * nfreq + freq_index for p in range(npol_fit)]))

    else:
        raise ValueError(
            f"Do not recognize polarisation {pol_fit}, "
            "possible values are 'XX', 'YY', 'I' or 'joint'"
        )

    pol = np.array(pol)
    x = np.zeros(nfreq * npol_fit, dtype=[("pol", "U8"), ("freq", np.float64)])
    for p, pstr in enumerate(np.atleast_1d(pol[ipol])):
        slc = slice(p * nfreq, (p + 1) * nfreq)
        x["pol"][slc] = pstr
        x["freq"][slc] = freq

    # Create results container
    results = containers.MCMCFit1D(
        x=x,
        freq=freq,
        pol=pol,
        mock=nmock,
        walker=nwalker,
        step=nsample,
        param=np.array(model.param_name),
        percentile=np.array(PERCENTILE),
    )

    results.attrs["seed"] = str(model.seed)
    results.attrs["model"] = model_name
    results.attrs["pol_fit"] = pol_fit

    results["mock"][:] = mock_stack
    results["stack"][:] = data_stack
    results["weight"][:] = weight_stack
    results["freq_flag"][:] = freq_flag

    if transfer is not None:
        results.add_dataset("transfer_function")
        results["transfer_function"][:] = transfer_stack

    if template is not None:
        results.add_dataset("template")
        results["template"][:] = template_stack

    results["cov"][:] = cov
    results["error"][:] = np.sqrt(np.diag(cov_flat).reshape(npol, nfreq))

    # Invert the covariance matrix to obtain the precision matrix
    Cinv = np.zeros_like(C)

    if flag_before:
        C = C[ifit][:, ifit]

    Cinvfit = np.linalg.pinv(C)

    if not flag_before:
        Cinvfit = Cinvfit[ifit][:, ifit]

    for ii, oi in enumerate(ifit):
        Cinv[oi, ifit] = Cinvfit[ii, :]

    results["precision"][:] = Cinv
    results["flag"][:] = np.diag(Cinv) > 0.0

    # Set the data for this polarisation
    y = data_stack[ipol]

    fit_kwargs = {"freq": freq, "data": y, "inv_cov": Cinv}
    eval_kwargs = {"freq": freq}

    if transfer is not None:
        fit_kwargs["transfer"] = transfer_stack[ipol]
        eval_kwargs["transfer"] = transfer_stack[:]
    else:
        fit_kwargs["transfer"] = None
        eval_kwargs["transfer"] = None

    if template is not None:
        fit_kwargs["template"] = template_stack[ipol]
        eval_kwargs["template"] = template_stack[:]

    model.set_data(**fit_kwargs)

    # Determine starting point for chains in parameter space
    pos = np.array([model.draw_random_parameters() for ww in range(nwalker)])

    nwalker, ndim = pos.shape

    # Create the sampler and run the MCMC
    sampler = emcee.EnsembleSampler(nwalker, ndim, model.log_probability)

    sampler.run_mcmc(pos, nsample, progress=False)

    chain = sampler.get_chain()

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
    results["autocorr_time"][:] = sampler.get_autocorr_time(quiet=True)
    results["acceptance_fraction"][:] = sampler.acceptance_fraction
    results["model_min_chisq"][:] = model.model(theta_min, **eval_kwargs)

    # Discard burn in and thin the chains
    flat_samples = results.samples(flat=True)

    # Compute percentiles of the posterior distribution
    q = np.percentile(flat_samples, PERCENTILE, axis=0).T
    dq = np.diff(q, axis=1)

    results["percentile"][:] = q
    results["median"][:] = q[:, 2]
    results["span_lower"][:] = dq[:, 1]
    results["span_upper"][:] = dq[:, 2]

    # Compute percentiles of the model
    mdl = np.zeros((flat_samples.shape[0], npol, nfreq), dtype=np.float32)
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
    
    See the arguments of the run_mcmc method for a list of attributes
    and their default values.
    """

    data = config.Property(proptype=str)
    mocks = config.Property(proptype=_list_or_glob)
    transfer = config.Property(proptype=str)
    template = config.Property(proptype=_list_or_glob)

    pol_fit = config.Property(proptype=str)
    model_name = config.Property(proptype=str)
    scale = config.Property(proptype=float)

    nwalker = config.Property(proptype=int)
    nsample = config.Property(proptype=int)

    max_freq = config.Property(proptype=float)
    flag_before = config.Property(proptype=bool)
    normalize_template = config.Property(proptype=bool)
    mean_subtract = config.Property(proptype=bool)
    recompute_weight = config.Property(proptype=bool)

    prior_spec = config.Property(proptype=dict)
    seed = config.Property(proptype=int)
    
    def setup(self):
        """Prepare all arguments to the run_mcmc function."""

        # Use the default values from the run_mcmc method,
        # so we do not have to repeat them in two places.
        defaults =  {k: v.default if v.default is not inspect.Parameter.empty else None
                     for k, v in signature.parameters.items()}
                     
        self.kwargs = {}
        for key, default_val in defaults.items():
            if hasattr(self, key):
                prop_val = getattr(self, key)
                self.kwargs[key] = prop_val if prop_val is not None else default_val
            else:
                self.log.warning("RunMCMC does not have a property corresponding "
                                 f"to the {key} keyword argument to run_mcmc.")

    def process(self):
        """Fit a model to the source stack using a MCMC."""

        result = run_mcmc(**self.kwargs)

        return result

