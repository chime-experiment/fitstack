"""Perform an MCMC fit of a model to a source stack."""

import numpy as np
import emcee

from draco.util import tools

from . import containers
from . import utils
from . import models

GUESS = {'amp': 1.0,
         'scale': 0.7,
         'offset': 0.0}

MODEL_LOOKUP = {"exponential": models.Exponential,
                "scaled_shifted_template": models.SST}

PERCENTILE = [2.5, 16, 50, 84, 97.5]


# main script
def process_data_mcmc(data, mocks, transfer=None, template=None, use_weight=False, invert_cov=True,
                      model_name='exp_model', scale=1e6, nwalker=32, nsample=15000,
                      max_freq=None, flag_before=True, normalize_template=False, mean_subtract=True, quiet=False):
    """Main routine for performing an MCMC fit to the stacked data.

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
    use_weight : bool
        Assume a diagonal covariance matrix with elements equal to the inverse
        of the weight dataset in the data container.  Useful if you do not yet
        have stacks on random mock catalogs to calculate the sample covariance.
        Default is False.
    invert_cov : bool
        Calculate the pseudo-inverse of the sample covariance matrix and use
        that as the precision matrix when evaluating the likelihood function.
        If this is False, then the precision matrix is assumed to be diagonal
        and is set to the inverse of the diagonal elements of the sample
        covariance matrix.  Default is True.
    model_name : {"exponential"|"scaled_shifted_template"}
        The name of the model to fit.
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
        Only relevant if max_freq is not None.  The frequecy offset flag will
        be applied prior to calculating the inverse of the covariance matrix.
    normalize_template : bool
        Divide the template by it's maximum value prior to fitting.
        Default is False.
    mean_subtract : bool
        Subtract the sample mean of the mocks from the data prior to fitting.
        Default is True.
    quiet : bool
        Suppress messages that report the progress.  Default is False.

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

    if isinstance(data, str):
        data = containers.FrequencyStackByPol.from_file(data)

    if transfer is not None and isinstance(transfer, str):
        transfer = containers.FrequencyStackByPol.from_file(transfer)
    if template is not None and isinstance(template, str):
        template = containers.FrequencyStackByPol.from_file(template)

    mocks = utils.load_mocks(mocks)
    nmocks = mocks.index_map['mock'].size

    freq = data.freq[:]
    nfreq = freq.size

    if model_name == "delta_model":

        model = MODEL_LOOKUP["scaled_shifted_template"]()

        template = containers.FrequencyStackByPol(attrs_from=data, axes_from=data)
        icenter = np.argmin(np.abs(freq))
        template['stack'][:] = 0.0
        template['stack'][:, icenter] = 1.0
        template['weight'][:] = 1.0

    else:

        model = MODEL_LOOKUP[model_name]()

    param_name = model.param_name
    nparam = len(param_name)

    # If a transfer function was not provided,
    # then we set it to the delta function.
    if transfer is None:

        transfer = containers.FrequencyStackByPol(attrs_from=data, axes_from=data)
        icenter = np.argmin(np.abs(freq))
        transfer['stack'][:] = 0.0
        transfer['stack'][:, icenter] = 1.0
        transfer['weight'][:] = 1.0

    # Determine the frequencies to fit
    if max_freq is not None:
        fit_flag = (np.abs(freq) < max_freq).astype(np.float32)
    else:
        fit_flag = np.ones(nfreq, dtype=np.float32)

    fit_index = np.flatnonzero(fit_flag)
    fit_slice = slice(fit_index[0], fit_index[-1] + 1)

    ndata = fit_index.size

    # Sort the frequency axis
    isort = np.argsort(freq)
    x = freq[isort]

    # Initialize all arrays
    data_stack, weight_stack = utils.initialize_pol(data)
    data_stack = scale * data_stack[..., isort]
    weight_stack = weight_stack[..., isort] / scale**2

    mock_stack = scale * mocks.stack[..., isort]

    transfer_stack, _ = utils.initialize_pol(transfer)
    transfer_stack = transfer_stack[..., isort]

    if template is not None:
        if len(template.pol) < 3:
            template_stack, _ = utils.initialize_pol(template)
        else:
            template_stack = template.stack[:]
        template_stack = scale * template_stack[..., isort]

    # Subtract mean value of mocks
    if mean_subtract:
        print("Subtracting the mean value of the mocks.")
        mu = np.mean(mock_stack, axis=0)
        mock_stack = mock_stack - mu[np.newaxis, ...]
        data_stack = data_stack - mu

    # Create results container
    pol = mocks.pol[:]
    npol = pol.size

    results = containers.MCMCFit1D(freq=x, pol=pol, mock=nmocks, walker=nwalker, step=nsample,
                                   param=np.array(model.param_name), percentile=np.array(PERCENTILE))

    results['mock'][:] = mock_stack
    results['stack'][:] = data_stack
    results['weight'][:] = weight_stack
    results['transfer_function'][:] = transfer_stack
    results['flag'][:] = fit_flag
    if template is not None:
        results.add_dataset("template")
        results['template'][:] = template_stack

    chisq = results['chisq'][:].view(np.ndarray)

    for pp, pstr in enumerate(pol):

        if not quiet:
            print(f"Fitting polarisation {pstr}")

        # Determine the covariance matrix
        Cfull = utils.covariance(mock_stack[:, pp], corr=False)

        if flag_before:
            C = Cfull[fit_slice, fit_slice]
        else:
            C = Cfull

        if use_weight:
            Cinvfit = np.diag(weight_stack[pp, fit_slice])
        elif invert_cov:
            Cinvfit = np.linalg.pinv(C)
        else:
            Cinvfit = np.diag(tools.invert_no_zero(np.diag(C)))

        if not flag_before and not use_weight:
            Cinvfit = Cinvfit[fit_slice, fit_slice]

        # Invert the covariance matrix
        Cinv = np.zeros_like(Cfull)
        Cinv[fit_slice, fit_slice] = Cinvfit

        # Save the covariance matrix to the ouput container
        results['cov'][pp] = Cfull
        results['inv_cov'][pp] = Cinv

        results['error'][pp] = np.sqrt(np.diag(Cfull))
        results['errori'][pp] = tools.invert_no_zero(np.sqrt(np.diag(Cinv)))

        # Set the data for this polarisation
        h = transfer_stack[pp]
        y = data_stack[pp]

        kwargs = {"data": y, "inv_cov": Cinv, "freq": x, "transfer": h}

        if model_name in ['scaled_shifted_template', 'delta_model']:

            t = template_stack[pp]
            if normalize_template:
                t = t / np.max(t)

            kwargs["template"] = t

        model.set_data(**kwargs)

        # Determine starting point for chains in parameter space
        pos = (0.05 * np.random.randn(nwalker, nparam) *
               np.array([model.boundary[pn][1] - model.boundary[pn][0] for pn in param_name]).reshape(1, nparam))
        pos += np.array([GUESS[pn] for pn in param_name]).reshape(1, nparam)

        nwalkers, ndim = pos.shape

        # Create the sampler and run the MCMC
        sampler = emcee.EnsembleSampler(nwalkers, ndim, model.log_probability)

        sampler.run_mcmc(pos, nsample, progress=False)

        chain = sampler.get_chain()

        # Compute the chisq
        for ss in range(nsample):
            for ww in range(nwalkers):
                chisq[pp, ss, ww] = -2.0 * model.log_likelihood(chain[ss, ww])

        # Find the minimum chisq
        imin = np.unravel_index(np.argmin(chisq[pp]), (nsample, nwalkers))

        theta_min = chain[imin]

        # Save the results to the output container
        results['chain'][pp] = chain
        results['autocorr_time'][pp] = sampler.get_autocorr_time(quiet=quiet)
        results['acceptance_fraction'][pp] = sampler.acceptance_fraction
        results['model_min_chisq'][pp] = model.model(theta_min)

        # Discard burn in and thin the chains
        flat_samples = results.samples(pp, flat=True)

        # Compute percentiles of the posterior distribution
        q = np.percentile(flat_samples, PERCENTILE, axis=0).T
        dq = np.diff(q, axis=1)

        mdl = np.zeros((flat_samples.shape[0], nfreq), dtype=np.float32)
        for ss, theta in enumerate(flat_samples):
            mdl[ss] = model.model(theta)

        results['percentile'][pp] = q
        results['median'][pp] = q[:, 2]
        results['span_lower'][pp] = dq[:, 1]
        results['span_upper'][pp] = dq[:, 2]

        results['model_percentile'][pp] = np.percentile(mdl, PERCENTILE, axis=0).T


    return results

