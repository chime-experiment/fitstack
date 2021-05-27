import numpy as np

from ch_util import cal_utils
from ch_util import tools

from draco.core import containers


def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _apply_shift(arr, freq, shift):

    slc = (None,) * (arr.ndim - 1) + (slice(None),)

    df = np.abs(freq[1] - freq[0])
    tau = np.fft.rfftfreq(freq.size, d=df * 1e6) * 1e6

    phase_shift = np.exp(-2.0J * np.pi * tau * offset)

    return np.fft.irfft(np.fft.rfft(arr, axis=-1) * phase_shift, axis=-1)


def convolve_template(data, template, pp=None, weight=None, max_df=6.0, offset=None):

    ikeep = np.flatnonzero(np.abs(template.freq[:]) <= max_df)

    if pp is None:
        d, w = combine_pol(data)
        w = weight if weight is not None else w
        t, wt = combine_pol(template)
    else:
        d = data['stack'][pp]
        w = weight if weight is not None else data['weight'][pp]
        t = template['stack'][pp].copy()

    if (offset is not None) and (offset != 0.0):
        t = _apply_shift(t, template.freq[:], offset)

    t = t[ikeep]
    t = t / np.max(t)

    nd = d.size
    nt = t.size

    ne = nt // 2

    no_edge_slc = slice(ne, nd - ne)
    afreq = data.freq[no_edge_slc]

    de = np.concatenate((np.zeros(ne, dtype=d.dtype),
                         d,
                         np.zeros(ne, dtype=d.dtype)))

    we = np.concatenate((np.zeros(ne, dtype=w.dtype),
                         w,
                         np.zeros(ne, dtype=w.dtype)))


    der = _rolling_window(de, nt)
    wer = _rolling_window(we, nt)

    norm = np.sum(wer * t[np.newaxis, :]**2, axis=-1)

    a = np.sum(wer * der * t[np.newaxis, :], axis=-1) * tools.invert_no_zero(norm)

#     vara = np.sum((wer * t[np.newaxis, :] * sig)**2, axis=-1) * tools.invert_no_zero(norm**2)
    vara = tools.invert_no_zero(norm)

    return afreq, a[no_edge_slc], np.sqrt(vara[no_edge_slc])


def process_data_matched_filter(data, mocks, template, scale=1e6, max_freq_sep=6.0,
                                scale_thermal=False, bins=60, brng=[-30, 30],
                                srng=[-0.8, 0.8], bnw=True, offsets=None):

    if isinstance(data, str):
        data = containers.FrequencyStackByPol.from_file(data)
        template = containers.FrequencyStackByPol.from_file(template)
        mocks = [containers.FrequencyStackByPol.from_file(mck) for mck in mocks]

    nmocks = len(mocks)

    if offsets is None:
        offsets = np.zeros(1, dtype=np.float32)
    noffset = len(offsets)

    freq = data.freq[:]
    nfreq = freq.size

    pol = np.append(data.pol[:], 'I')
    npol = pol.size
    mpol = npol - 1

    # Load the mocks into an array
    mock_stack = np.zeros((npol, nmocks, nfreq), dtype=np.float32)
    mock_weight = np.zeros((npol, nmocks, nfreq), dtype=np.float32)

    for mm, mck in enumerate(mocks):
        mock_stack[0:mpol, mm, :] = mck['stack'][:]
        mock_weight[0:mpol, mm, :] = mck['weight'][:]

        ms, mw = combine_pol(mck)
        mock_stack[-1, mm, :] = ms
        mock_weight[-1, mm, :] = mw

    # Determine weight as a function of frequency offset
    obs_std_mock = np.std(mock_stack, axis=1)

    if scale_thermal:
        exp_std_mock = np.mean(np.sqrt(tools.invert_no_zero(mock_weight)), axis=1)
        noise_scale_factor = np.median(obs_std_mock * tools.invert_no_zero(exp_std_mock), axis=-1)
        weight = tools.invert_no_zero((noise_scale_factor[:, np.newaxis] * exp_std_mock)**2)
    else:
        weight = tools.invert_no_zero(obs_std_mock ** 2)

    # Calculate the standard deviation of the mocks by fitting to histogram
    dcmock = {}
    for pp, pstr in enumerate(pol):
        dcmock[pstr] = cal_utils.fit_histogram(scale * mock_stack[pp].flatten(), bins=bins, rng=brng,
                                               no_weight=bnw, test_normal=True, return_histogram=True)

    # Convolve data and mocks with template
    for pp in range(npol):

        pval = None if pp >= mpol else pp
        wval = weight[pp]

        for oo, off in enumerate(offsets):

            adf, app, eapp = convolve_template(data, template, pp=pval, weight=wval,
                                               max_df=max_freq_sep, offset=off)

            if not pp and not oo:
                amp = np.zeros((npol, noffset, app.size), dtype=np.float32)
                err_amp = np.zeros((npol, noffset, app.size), dtype=np.float32)

            amp[pp, oo] = app
            err_amp[pp, oo] = eapp

    for mm, mck in enumerate(mocks):

        for pp in range(npol):

            pval = None if pp >= mpol else pp
            wval = weight[pp]

            for oo, off in enumerate(offsets):

                madf, ma, ema = convolve_template(mck, template, pp=pval, weight=wval,
                                                  max_df=max_freq_sep, offset=off)

                if not mm and not pp and not oo:
                    mamp = np.zeros((npol, noffset, nmocks, madf.size), dtype=np.float32)
                    err_mamp = np.zeros((npol, noffset, nmocks, madf.size), dtype=np.float32)

                mamp[pp, oo, mm, :] = ma
                err_mamp[pp, oo, mm, :] = ema

    # Calculate the standard deviation of the convolved template by fitting to histogram
    dcmamp = {}
    for pp, pstr in enumerate(pol):
        dcmamp[pstr] = cal_utils.fit_histogram(scale * mamp[pp].flatten(), bins=bins, rng=brng,
                                               no_weight=bnw, test_normal=True, return_histogram=True)

    # Package results
    results = {}
    results['offset'] = offsets
    results['freq'] = freq
    results['freq_convolved'] = adf
    results['pol'] = pol

    results['mock'] = mock_stack
    results['mock_convolved'] = mamp

    results['distribution_mock'] = dcmock
    results['distribution_mock_convolved'] = dcmamp

    dd, dw = combine_pol(data)
    td, tw = combine_pol(template)

    results['data'] = np.concatenate((data['stack'][:], dd[np.newaxis, :]), axis=0)
    results['template'] = np.concatenate((template['stack'][:], td[np.newaxis, :]), axis=0)

    results['sigma'] = np.array([dcmock[pstr]['par'][2] for pstr in pol])
    results['sigma_convolved'] = np.array([dcmamp[pstr]['par'][2] for pstr in pol])

    for key in ['peak_index', 'peak_freq', 'peak_value',
                'peak_index_convolved', 'peak_freq_convolved', 'peak_value_convolved']:
        dtype = np.int if 'index' in key else np.float
        if key == 'peak_index_convolved':
            results[key] = np.zeros((npol, 2), dtype=dtype)
        else:
            results[key] = np.zeros(npol, dtype=dtype)

    isrch = np.flatnonzero((freq >= srng[0]) & (freq <= srng[1]))
    isrchc = np.flatnonzero((adf >= srng[0]) & (adf <= srng[1]))

    for pp in range(npol):

        imax = isrch[np.argmax(results['data'][pp][isrch])]

        results['peak_index'][pp] = imax
        results['peak_freq'][pp] = freq[imax]
        results['peak_value'][pp] = scale * results['data'][pp][imax]

        imaxc = np.zeros(noffset, dtype=np.int)
        maxc = np.zeros(noffset, dtype=np.float64)
        for oo in range(noffset):
            imaxc[oo] = isrchc[np.argmax(amp[pp, oo, isrchc])]
            maxc[oo] = amp[pp, oo, imaxc[oo]]

        imaxo = np.argmax(maxc)
        imaxc = imaxc[imaxo]
        maxc = maxc[imaxo]

        results['peak_index_convolved'][pp] = np.array([imaxo, imaxc])
        results['peak_freq_convolved'][pp] = adf[imaxc] + offsets[imaxo]
        results['peak_value_convolved'][pp] = scale * amp[pp, imaxo, imaxc]

    results['s2n'] = results['peak_value'] * tools.invert_no_zero(results['sigma'])
    results['s2n_convolved'] = results['peak_value_convolved'] * tools.invert_no_zero(results['sigma_convolved'])

    return results
