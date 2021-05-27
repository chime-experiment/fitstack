import numpy as np
import emcee

from ch_util import tools

from . import containers
from . import utils
from . import models

BOUNDARY = {'amp':    [0.0, 500.0],
            'scale':  [0.1, 10.0],
            'offset': [-1.0, 1.0]}

GUESS = {'amp': 125.0,
         'scale': 0.7,
         'offset': 0.0}


# main script
def process_data_mcmc(data, mocks, transfer, template=None, invert_cov=True,
                      model_name='exp_model', scale=1e6, nwalker=32, nsample=5000,
                      max_freq=6.0, flag_before=True, quiet=True):

    if isinstance(data, str):
        data = containers.FrequencyStackByPol.from_file(data)
        transfer = containers.FrequencyStackByPol.from_file(transfer)
        mocks = load_mocks(mocks)
        if template is not None:
            template = containers.FrequencyStackByPol.from_file(template)

    pc = [2.5, 16, 50, 84, 97.5]
    nperc = len(pc)

    nmocks = mocks.index_map['mock'].size

    freq = data.freq[:]
    nfreq = freq.size

    if model_name == 'exp_model':

        param_name = ['amp', 'scale', 'offset']
        model = models.exp_model
        prob = models.exp_model_log_probability
        likelihood = models.exp_model_log_likelihood

    elif model_name == 'scaled_shifted_template':

        param_name = ['amp', 'offset']
        model = models.sst_model
        prob = models.sst_model_log_probability
        likelihood = models.sst_model_log_likelihood

    elif model_name == 'delta_model':

        param_name = ['amp', 'offset']
        model = models.sst_model
        prob = models.sst_model_log_probability
        likelihood = models.sst_model_log_likelihood

        template = containers.FrequencyStackByPol(attrs_from=data, axes_from=data)
        icenter = np.argmin(np.abs(freq))
        template['stack'][:] = 0.0
        template['stack'][:, icenter] = 1.0
        template['weight'][:] = 1.0

    else:
        raise ValueError("Do not recognize model %s" % model_name)

    nparam = len(param_name)

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
    data_stack = scale * utils.initialize_pol(data)[..., isort]

    mock_stack = scale * mocks.stack[..., isort]

    transfer_stack = utils.initialize_pol(transfer)[..., isort]

    if template is not None:
        template_stack = scale * utils.initialize_pol(template)[..., isort]

    # Create results dictionary
    pol = mocks.pol[:]
    npol = pol.size

    results = {}
    results['pol'] = pol
    results['freq'] = x
    results['param'] = param_name
    results['ndof'] = ndata - nparam

    results['flag'] = fit_flag
    results['mock'] = mock_stack
    results['data'] = data_stack
    results['transfer'] = transfer_stack
    if template is not None:
        results['template'] = template_stack

    results['err'] = np.zeros((npol, nfreq), dtype=np.float64)
    results['ierr'] = np.zeros((npol, nfreq), dtype=np.float64)
    results['cov'] = np.zeros((npol, nfreq, nfreq), dtype=np.float64)
    results['inv_cov'] = np.zeros((npol, nfreq, nfreq), dtype=np.float64)

    results['chain'] = {}
    results['samples'] = {}
    results['tau'] = np.zeros((npol, nparam), dtype=np.int)

    results['median'] = np.zeros((npol, nparam), dtype=np.float32)
    results['span_lower'] = np.zeros((npol, nparam), dtype=np.float32)
    results['span_upper'] = np.zeros((npol, nparam), dtype=np.float32)

    results['model_percentile'] = np.zeros((npol, nperc, nfreq), dtype=np.float32)

    results['chisq'] = {}
    results['min_chisq'] = np.zeros((npol,), dtype=np.float32)

    for pp, pstr in enumerate(pol):

        Cfull = utils.covariance(mock_stack[:, pp], corr=False)

        if flag_before:
            C = Cfull[fit_slice, fit_slice]
        else:
            C = Cfull

        if invert_cov:
            Cinvfit = np.linalg.pinv(C)
        else:
            Cinvfit = np.diag(tools.invert_no_zero(np.diag(C)))

        if not flag_before:
            Cinvfit = Cinvfit[fit_slice, fit_slice]

        Cinv = np.zeros_like(Cfull)
        Cinv[fit_slice, fit_slice] = Cinvfit

        results['cov'][pp] = Cfull
        results['inv_cov'][pp] = Cinv

        results['err'][pp] = np.sqrt(np.diag(Cfull))
        results['ierr'][pp] = tools.invert_no_zero(np.sqrt(np.diag(Cinv)))

        h = transfer_stack[pp]

        y = data_stack[pp]

        if model_name == 'exp_model':
            aux = (x, h)

        elif model_name in ['scaled_shifted_template', 'delta_model']:

            t = template_stack[pp]
            t = t / np.max(t)

            aux = (x, h, t)

        else:
            raise ValueError("Do not recognize model %s" % model_name)

        args = (param_name, y, Cinv) + aux

        # pos = (np.random.uniform(size=nwalker * nparam).reshape(nwalker, nparam) *
        #        np.array([BOUNDARY[pn][1] - BOUNDARY[pn][0] for pn in param_name]).reshape(1, nparam))
        # pos += np.array([BOUNDARY[pn][0] for pn in param_name]).reshape(1, nparam)

        pos = (0.05 * np.random.randn(nwalker, nparam) *
               np.array([BOUNDARY[pn][1] - BOUNDARY[pn][0] for pn in param_name]).reshape(1, nparam))
        pos += np.array([GUESS[pn] for pn in param_name]).reshape(1, nparam)

        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(nwalkers, ndim, prob, args=args)
        sampler.run_mcmc(pos, nsample, progress=False)

        results['chain'][pstr] = sampler.get_chain()

        tau = sampler.get_autocorr_time(quiet=quiet)
        results['tau'][pp] = np.array(tau)

        thin = np.max([int(tt) // 2 for tt in tau])
        discard = np.max([int(tt) * 10 for tt in tau])

        flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        results['samples'][pstr] = flat_samples

        print("Discarding %d samples as burn in.  Thinning by %d samples.  %d total samples remain." %
             (discard, thin, flat_samples.size))

        q = np.percentile(flat_samples, [16, 50, 84], axis=0)
        dq = np.diff(q, axis=0)

        nsamples = flat_samples.shape[0]

        mdl = np.zeros((nsamples, nfreq), dtype=np.float32)
        chisq = np.zeros((nsamples,), dtype=np.float32)
        for ss in range(nsamples):
            mdl[ss, :] = model(flat_samples[ss], *aux)
            chisq[ss] = -2.0 * likelihood(flat_samples[ss], *args[1:])

        results['median'][pp] = q[1, :]
        results['span_lower'][pp] = dq[0, :]
        results['span_upper'][pp] = dq[1, :]

        results['model_percentile'][pp] = np.percentile(mdl, pc, axis=0)

        results['chisq'][pstr] = chisq
        results['min_chisq'][pp] = np.min(chisq)


    return results

