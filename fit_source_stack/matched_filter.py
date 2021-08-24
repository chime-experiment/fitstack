"""Early version of the analysis that used a matched filter approach.

Both the data and mocks are convolved with a template.  The distribution
of values observed in the convolved mocks is fit to a Gaussian and used to 
determine the significance of the largest excursion observed in the convolved data.
"""

import numpy as np
import scipy.stats
from scipy.optimize import curve_fit

from draco.util import tools
from draco.core import containers


def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _apply_shift(arr, freq, shift):

    slc = (None,) * (arr.ndim - 1) + (slice(None),)

    df = np.abs(freq[1] - freq[0])
    tau = np.fft.rfftfreq(freq.size, d=df * 1e6) * 1e6

    phase_shift = np.exp(-2.0j * np.pi * tau * offset)

    return np.fft.irfft(np.fft.rfft(arr, axis=-1) * phase_shift, axis=-1)


def fit_histogram(
    arr,
    bins="auto",
    rng=None,
    no_weight=False,
    test_normal=False,
    return_histogram=False,
):
    """
    Fit a gaussian to a histogram of the data.

    Parameters
    ----------
    arr : np.ndarray
        1D array containing the data.
        Arrays with more than one dimension are flattened.
    bins : int or sequence of scalars or str
        - If `bins` is an int, it defines the number of equal-width bins in `rng`.
        - If `bins` is a sequence, it defines a monotonically increasing array of
          bin edges, including the rightmost edge, allowing for non-uniform bin widths.
        - If `bins` is a string, it defines a method for computing the bins.
    rng : (float, float)
        The lower and upper range of the bins.  If not provided, then the range spans
        the minimum to maximum value of `arr`.
    no_weight : bool
        Give equal weighting to each histogram bin.  Otherwise use proper weights
        based on number of counts observed in each bin.
    test_normal : bool
        Apply the Shapiro-Wilk and Anderson-Darling tests for normality to the data.
    return_histogram : bool
        Return the histogram.  Otherwise return only the best fit parameters and
        test statistics.

    Returns
    -------
    results: dict
        Dictionary containing the following fields:
    indmin : int
        Only bins whose index is greater than indmin were included in the fit.
    indmax : int
        Only bins whose index is less than indmax were included in the fit.
    xmin : float
        The data value corresponding to the centre of the `indmin` bin.
    xmax : float
        The data value corresponding to the centre of the `indmax` bin.
    par: [float, float, float]
        The parameters of the fit, ordered as [peak, mu, sigma].
    chisq: float
        The chi-squared of the fit.
    ndof : int
        The number of degrees of freedom of the fit.
    pte : float
        The probability to observe the chi-squared of the fit.

    If `return_histogram` is True, then `results` will also contain
    the following fields:

        bin_centre : np.ndarray
            The bin centre of the histogram.
        bin_count : np.ndarray
            The bin counts of the histogram.

    If `test_normal` is True, then `results` will also contain the following fields:

        shapiro : dict
            stat : float
                The Shapiro-Wilk test statistic.
            pte : float
                The probability to observe `stat` if the data were
                drawn from a gaussian.
        anderson : dict
            stat : float
                The Anderson-Darling test statistic.
            critical : list of float
                The critical values of the test statistic.
            alpha : list of float
                The significance levels corresponding to each critical value.
            past : list of bool
                Boolean indicating if the data passes the test for each critical value.
    """
    # Make sure the data is 1D
    data = np.ravel(arr)

    # Histogram the data
    count, xbin = np.histogram(data, bins=bins, range=rng)
    cbin = 0.5 * (xbin[0:-1] + xbin[1:])

    cbin = cbin.astype(np.float64)
    count = count.astype(np.float64)

    # Form initial guess at parameter values using median and MAD
    nparams = 3
    par0 = np.zeros(nparams, dtype=np.float64)
    par0[0] = np.max(count)
    par0[1] = np.median(data)
    par0[2] = 1.48625 * np.median(np.abs(data - par0[1]))

    # Find the first zero points on either side of the median
    cont = True
    indmin = np.argmin(np.abs(cbin - par0[1]))
    while cont:
        indmin -= 1
        cont = (count[indmin] > 0.0) and (indmin > 0)
    indmin += count[indmin] == 0.0

    cont = True
    indmax = np.argmin(np.abs(cbin - par0[1]))
    while cont:
        indmax += 1
        cont = (count[indmax] > 0.0) and (indmax < (len(count) - 1))
    indmax -= count[indmax] == 0.0

    # Restrict range of fit to between zero points
    x = cbin[indmin : indmax + 1]
    y = count[indmin : indmax + 1]
    yerr = np.sqrt(y * (1.0 - y / np.sum(y)))

    sigma = None if no_weight else yerr

    # Require positive values of amp and sigma
    bnd = (np.array([0.0, -np.inf, 0.0]), np.array([np.inf, np.inf, np.inf]))

    # Define the fitting function
    def gauss(x, peak, mu, sigma):
        return peak * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))

    # Perform the fit
    par, var_par = curve_fit(
        gauss,
        cbin[indmin : indmax + 1],
        count[indmin : indmax + 1],
        p0=par0,
        sigma=sigma,
        absolute_sigma=(not no_weight),
        bounds=bnd,
        method="trf",
    )

    # Calculate quality of fit
    chisq = np.sum(((y - gauss(x, *par)) / yerr) ** 2)
    ndof = np.size(y) - nparams
    pte = 1.0 - scipy.stats.chi2.cdf(chisq, ndof)

    # Store results in dictionary
    results_dict = {}
    results_dict["indmin"] = indmin
    results_dict["indmax"] = indmax
    results_dict["xmin"] = cbin[indmin]
    results_dict["xmax"] = cbin[indmax]
    results_dict["par"] = par
    results_dict["chisq"] = chisq
    results_dict["ndof"] = ndof
    results_dict["pte"] = pte

    if return_histogram:
        results_dict["bin_centre"] = cbin
        results_dict["bin_count"] = count

    # If requested, test normality of the main distribution
    if test_normal:
        flag = (data > cbin[indmin]) & (data < cbin[indmax])
        shap_stat, shap_pte = scipy.stats.shapiro(data[flag])

        results_dict["shapiro"] = {}
        results_dict["shapiro"]["stat"] = shap_stat
        results_dict["shapiro"]["pte"] = shap_pte

        ander_stat, ander_crit, ander_signif = scipy.stats.anderson(
            data[flag], dist="norm"
        )

        results_dict["anderson"] = {}
        results_dict["anderson"]["stat"] = ander_stat
        results_dict["anderson"]["critical"] = ander_crit
        results_dict["anderson"]["alpha"] = ander_signif
        results_dict["anderson"]["pass"] = ander_stat < ander_crit

    # Return dictionary
    return results_dict


def convolve_template(data, template, pp=None, weight=None, max_df=6.0, offset=None):

    ikeep = np.flatnonzero(np.abs(template.freq[:]) <= max_df)

    if pp is None:
        d, w = combine_pol(data)
        w = weight if weight is not None else w
        t, wt = combine_pol(template)
    else:
        d = data["stack"][pp]
        w = weight if weight is not None else data["weight"][pp]
        t = template["stack"][pp].copy()

    if (offset is not None) and (offset != 0.0):
        t = _apply_shift(t, template.freq[:], offset)

    t = t[ikeep]
    t = t / np.max(t)

    nd = d.size
    nt = t.size

    ne = nt // 2

    no_edge_slc = slice(ne, nd - ne)
    afreq = data.freq[no_edge_slc]

    de = np.concatenate((np.zeros(ne, dtype=d.dtype), d, np.zeros(ne, dtype=d.dtype)))

    we = np.concatenate((np.zeros(ne, dtype=w.dtype), w, np.zeros(ne, dtype=w.dtype)))

    der = _rolling_window(de, nt)
    wer = _rolling_window(we, nt)

    norm = np.sum(wer * t[np.newaxis, :] ** 2, axis=-1)

    a = np.sum(wer * der * t[np.newaxis, :], axis=-1) * tools.invert_no_zero(norm)

    #     vara = np.sum((wer * t[np.newaxis, :] * sig)**2, axis=-1) * tools.invert_no_zero(norm**2)
    vara = tools.invert_no_zero(norm)

    return afreq, a[no_edge_slc], np.sqrt(vara[no_edge_slc])


def process_data_matched_filter(
    data,
    mocks,
    template,
    scale=1e6,
    max_freq_sep=6.0,
    scale_thermal=False,
    bins=60,
    brng=[-30, 30],
    srng=[-0.8, 0.8],
    bnw=True,
    offsets=None,
):

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

    pol = np.append(data.pol[:], "I")
    npol = pol.size
    mpol = npol - 1

    # Load the mocks into an array
    mock_stack = np.zeros((npol, nmocks, nfreq), dtype=np.float32)
    mock_weight = np.zeros((npol, nmocks, nfreq), dtype=np.float32)

    for mm, mck in enumerate(mocks):
        mock_stack[0:mpol, mm, :] = mck["stack"][:]
        mock_weight[0:mpol, mm, :] = mck["weight"][:]

        ms, mw = combine_pol(mck)
        mock_stack[-1, mm, :] = ms
        mock_weight[-1, mm, :] = mw

    # Determine weight as a function of frequency offset
    obs_std_mock = np.std(mock_stack, axis=1)

    if scale_thermal:
        exp_std_mock = np.mean(np.sqrt(tools.invert_no_zero(mock_weight)), axis=1)
        noise_scale_factor = np.median(
            obs_std_mock * tools.invert_no_zero(exp_std_mock), axis=-1
        )
        weight = tools.invert_no_zero(
            (noise_scale_factor[:, np.newaxis] * exp_std_mock) ** 2
        )
    else:
        weight = tools.invert_no_zero(obs_std_mock ** 2)

    # Calculate the standard deviation of the mocks by fitting to histogram
    dcmock = {}
    for pp, pstr in enumerate(pol):
        dcmock[pstr] = cal_utils.fit_histogram(
            scale * mock_stack[pp].flatten(),
            bins=bins,
            rng=brng,
            no_weight=bnw,
            test_normal=True,
            return_histogram=True,
        )

    # Convolve data and mocks with template
    for pp in range(npol):

        pval = None if pp >= mpol else pp
        wval = weight[pp]

        for oo, off in enumerate(offsets):

            adf, app, eapp = convolve_template(
                data, template, pp=pval, weight=wval, max_df=max_freq_sep, offset=off
            )

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

                madf, ma, ema = convolve_template(
                    mck, template, pp=pval, weight=wval, max_df=max_freq_sep, offset=off
                )

                if not mm and not pp and not oo:
                    mamp = np.zeros(
                        (npol, noffset, nmocks, madf.size), dtype=np.float32
                    )
                    err_mamp = np.zeros(
                        (npol, noffset, nmocks, madf.size), dtype=np.float32
                    )

                mamp[pp, oo, mm, :] = ma
                err_mamp[pp, oo, mm, :] = ema

    # Calculate the standard deviation of the convolved template by fitting to histogram
    dcmamp = {}
    for pp, pstr in enumerate(pol):
        dcmamp[pstr] = cal_utils.fit_histogram(
            scale * mamp[pp].flatten(),
            bins=bins,
            rng=brng,
            no_weight=bnw,
            test_normal=True,
            return_histogram=True,
        )

    # Package results
    results = {}
    results["offset"] = offsets
    results["freq"] = freq
    results["freq_convolved"] = adf
    results["pol"] = pol

    results["mock"] = mock_stack
    results["mock_convolved"] = mamp

    results["distribution_mock"] = dcmock
    results["distribution_mock_convolved"] = dcmamp

    dd, dw = combine_pol(data)
    td, tw = combine_pol(template)

    results["data"] = np.concatenate((data["stack"][:], dd[np.newaxis, :]), axis=0)
    results["template"] = np.concatenate(
        (template["stack"][:], td[np.newaxis, :]), axis=0
    )

    results["sigma"] = np.array([dcmock[pstr]["par"][2] for pstr in pol])
    results["sigma_convolved"] = np.array([dcmamp[pstr]["par"][2] for pstr in pol])

    for key in [
        "peak_index",
        "peak_freq",
        "peak_value",
        "peak_index_convolved",
        "peak_freq_convolved",
        "peak_value_convolved",
    ]:
        dtype = np.int if "index" in key else np.float
        if key == "peak_index_convolved":
            results[key] = np.zeros((npol, 2), dtype=dtype)
        else:
            results[key] = np.zeros(npol, dtype=dtype)

    isrch = np.flatnonzero((freq >= srng[0]) & (freq <= srng[1]))
    isrchc = np.flatnonzero((adf >= srng[0]) & (adf <= srng[1]))

    for pp in range(npol):

        imax = isrch[np.argmax(results["data"][pp][isrch])]

        results["peak_index"][pp] = imax
        results["peak_freq"][pp] = freq[imax]
        results["peak_value"][pp] = scale * results["data"][pp][imax]

        imaxc = np.zeros(noffset, dtype=np.int)
        maxc = np.zeros(noffset, dtype=np.float64)
        for oo in range(noffset):
            imaxc[oo] = isrchc[np.argmax(amp[pp, oo, isrchc])]
            maxc[oo] = amp[pp, oo, imaxc[oo]]

        imaxo = np.argmax(maxc)
        imaxc = imaxc[imaxo]
        maxc = maxc[imaxo]

        results["peak_index_convolved"][pp] = np.array([imaxo, imaxc])
        results["peak_freq_convolved"][pp] = adf[imaxc] + offsets[imaxo]
        results["peak_value_convolved"][pp] = scale * amp[pp, imaxo, imaxc]

    results["s2n"] = results["peak_value"] * tools.invert_no_zero(results["sigma"])
    results["s2n_convolved"] = results["peak_value_convolved"] * tools.invert_no_zero(
        results["sigma_convolved"]
    )

    return results
