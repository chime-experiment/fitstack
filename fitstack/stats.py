import logging

import numpy as np
import scipy.optimize
import scipy.stats

from statsmodels.stats.diagnostic import anderson_statistic


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


def F_scaling(n, p, p_eff):
    """Compute chi^2-to-F scaling factor:

    From Sellentin+Heavens 2015 (arXiv:1511.05969), a chi^2
    value computed with an estimated covariance will follow
    an F distribution if the value is multiplied by

    .. math::

        (n - p + 1) / (n p) .

    If this is used in the context of fitting a model with
    degenerate parameters, the F distribution may not be
    exact, so we allow for an effective number of degrees
    of freedom by allowing the `p` in the denominator to
    differ from the one in the numerator.

    Parameters
    ----------
    n, p, p_eff : float
        Parameters in the rescaling factor.

    Returns
    -------
    factor : float
        Rescaling factor.
    """
    return (n - p + 1) / (n * p_eff)


def fit_chi2_to_array(vals):
    """Fit a chi^2 distribution to a set of values.

    Parameters
    ----------
    vals : np.ndarray
        Values to fit to.

    Returns
    -------
    ndof : float
        Best-fit number of degrees of freedom.
    """
    return scipy.stats.chi2.fit(vals, floc=0, fscale=1)[0]


def fit_F_to_scaled_array(vals, n, p, p_eff_0=None, eps=1e-8):
    """Fit an F distribution to a rescaled set of values.

    We take the input values, rescale by

    .. math::

        (n - p + 1) / (n p_{eff}) ,

    and then fit for :math:`p_eff` by maximizing the likelihood that
    the rescaled values are drawn from an :math:`F_{p_{eff}, n - p + 1}`
    distribution.

    Parameters
    ----------
    vals : np.ndarray
        Values to fit to.
    n, p : float
        Parameters of the target `F` distribution. If the input values
        are :math:`\chi^2` or :math:`\Delta\chi^2` values computed using a
        covariance matrix that was estimated from N simulations, one should
        specify :math:`n=N-1`. `p` should be the length of the data vector
    p_eff_0 : float, optional
        Initial guess for :math:`p_{eff}`. If not given, a :math:`\chi^2`
        distribution is fit to the (unscaled) values and the best-fit
        number of degrees of freedom is used. Default: None.
    eps : float, optional
        Step size used for numerical approximation of the Jacobian used
        in the L-BFGS-B optimizer. Default: 1e-8.

    Returns
    -------
    p_eff : float
        Best-fit :math:`p_{\rm eff}` value.
    resd : scipy.optimize.OptimizeResult
        Full fit results.
    """

    _MIN_P_EFF = 1e-6

    def _neg_log_likelihood(x):

        p_eff = x[0]

        # p_eff <=0 is impossible, so return infinite negative log likelihood
        # in that case
        if p_eff <= 0:
            return np.inf

        # Set scaling factor to rescale input values into F-distributed form,
        # and rescale values
        s = F_scaling(n, p, p_eff)
        vals_scaled = vals * s

        # Evaluate pdf of proposed F distribution at scaled values
        pdf_vals = scipy.stats.f.pdf(vals_scaled, p_eff, n - p + 1)

        # If zeros or infinities are detected, return infinite negative log
        # likelihood
        if np.any(pdf_vals <= 0) or not np.isfinite(pdf_vals).all():
            return np.inf

        # Return negative log likelihood, corresponding to probability
        # that given values are distributed like F_{p_eff, n - p + 1}
        return -np.sum(np.log(pdf_vals) + np.log(s))

    # If no initial guess for p_eff provided, set to best-fit number of d.o.f.
    # for a chi^2 distribution
    if p_eff_0 is None:
        p_eff_0 = fit_chi2_to_array(vals)

    # Fit for p_eff using L-BFGS-B
    resd = scipy.optimize.minimize(
        _neg_log_likelihood,
        x0=np.array([p_eff_0]),
        bounds=[(_MIN_P_EFF, None)],
        options={"eps": eps},
        method="L-BFGS-B",
    )

    # Return p_eff and full fit results
    p_eff = resd.x[0]
    return p_eff, resd


def compute_MC_calibrated_distribution_test(
    vals,
    test="KS",
    dist="chi2",
    seed=0,
    n_for_F=998,
    p_for_F=16,
    verbose=False,
    n_mc_sims=1000,
    bootstrap=False,
    return_ndof=False,
):
    """Compute Monte-Carlo-calibrated distribution test.

    For a set of values, we compute the specified test statistic
    (KS or AD) comparing the values to the specified distribution
    (chi^2 or F) with `n_dof` parameter fit to the values themselves.
    We then generate `n_mc_sims` draws of the same number of values
    from the best-fit distribution, re-fit the distribution to the draw,
    and compute the test statistic for each draw and the corresponding
    best-fit distribution. The number of draws with test statistic
    greater than the data test statistic is the p-value for the test,
    but calibrated via Monte Carlo. We also return the data test
    statistic and test statistics for each draw.

    Alternatively, this function allows one to draw bootstrap resamples
    of the input set of values instead of generating Monte Carlo
    draws from the fitted distribution.

    Parameters
    ----------
    vals : np.ndarray
        Array of values to compare to a distribution.
    test : str, optional
        Test to apply. Must be one of "KS" (Kolmogorov-Smirnov)
        or "AD" (Anderson-Darling). Default: "KS".
    dist : str, optional
        Distribution to test. Must be one of "chi2" or "F".
        Default: "chi2".
    seed : int, optional
        Seed for random number generator. Default: 0.
    n_for_F, p_for_F: float, optional
        n and p parameters for "df2" of the F distribution.
        The p parameter for "df1" will be fit separately from
        `p_for_F`. Defaults: 998, 16.
    verbose : bool, optional
        Use `tqdm` package to display progress bar. Default: False.
    n_mc_sims : int, optional
        Number of Monte Carlo realizations of distribution.
        Default: 1000.
    bootstrap : bool, optional
        Draw bootstrap resamples of original set of values, instead
        of drawing from best-fit distribution. Default: False.
    return_ndof : bool, optional
        Return array of fitted n_dof values for each Monte Carlo
        realization. Default: False.

    Returns
    -------
    p_cal : float
        Calibrated p-value corresponding to chosen test.
    statistic_data : float
        Test statistic evaluated on input values.
    statistic_sim : np.ndarray
        Array of test statistics computed via Monte Carlo.
    ndof_sim : np.ndarray
        Array of fitted n_dof values for each Monte Carlo realization.
        Only returned if `return_dof` is True.
    """

    _TESTS = ["KS", "AD"]
    _DISTS = ["chi2", "F"]

    # Ensure that valid test statistic and distribution are being requested
    if test not in _TESTS:
        raise NotImplementedError(
            f"Test {test} not implemented (must be one of {_TESTS})"
        )

    if dist not in _DISTS:
        raise NotImplementedError(
            f"Distribution {dist} not implemented (must be one of {_DISTS})"
        )

    # Fit specified distribution to input values, and compute desired
    # test statistic
    if dist == "chi2":
        ndof_data = fit_chi2_to_array(vals)

        if test == "KS":
            statistic_data, _ = scipy.stats.kstest(vals, "chi2", args=(ndof_data,))
        elif test == "AD":
            statistic_data = anderson_statistic(
                vals, scipy.stats.chi2, params=(ndof_data,)
            )

    elif dist == "F":
        ndof_data, _ = fit_F_to_scaled_array(vals, n_for_F, p_for_F)

        # Set second parameter for F distribution
        df_den = n_for_F - p_for_F + 1

        # Note that input values need to be scaled before comparing to
        # F distribution
        if test == "KS":
            statistic_data, _ = scipy.stats.kstest(
                vals * F_scaling(n_for_F, p_for_F, ndof_data),
                "f",
                args=(ndof_data, df_den),
            )
        elif test == "AD":
            statistic_data = anderson_statistic(
                vals * F_scaling(n_for_F, p_for_F, ndof_data),
                scipy.stats.f,
                params=(ndof_data, df_den),
            )

    # Define lists to store fit parameters and test statistics for
    # each Monte Carlo simulation
    ndof_sim, statistic_sim = [], []

    # Initialize random number generator
    rng = np.random.default_rng(seed=seed)

    # Get size of dataset
    n_mocks = len(vals)

    # Use tqdm to print status during loop over Monte Carlo simulations,
    # if desired
    if verbose:
        s_range = trange(n_mc_sims)
    else:
        s_range = range(n_mc_sims)

    # Loop over MC sims
    for s in s_range:
        if dist == "chi2":
            # Generate n_mocks random draws from chi^2 distribution
            # or input values, and fit n_dof
            if bootstrap:
                sim_vals = rng.choice(vals, size=len(vals), replace=True)
            else:
                sim_vals = rng.chisquare(df=ndof_data, size=n_mocks)

            ndof = fit_chi2_to_array(sim_vals)

            if test == "KS":
                statistic, _ = scipy.stats.kstest(sim_vals, "chi2", args=(ndof,))
            elif test == "AD":
                statistic = anderson_statistic(
                    sim_vals, scipy.stats.chi2, params=(ndof,)
                )

        elif dist == "F":
            # Generate n_mocks random draws from F distribution
            # or input values, and fit n_dof
            if bootstrap:
                sim_vals = rng.choice(
                    vals,
                    size=len(vals),
                    replace=True,
                )
                ndof, _ = fit_F_to_scaled_array(sim_vals, n_for_F, p_for_F)
                sim_vals *= F_scaling(n_for_F, p_for_F, ndof)
            else:
                sim_vals = rng.f(dfnum=ndof_data, dfden=df_den, size=n_mocks)
                ndof = scipy.stats.f.fit(sim_vals, fdfd=df_den, floc=0, fscale=1)[0]

            if test == "KS":
                statistic, _ = scipy.stats.kstest(sim_vals, "f", args=(ndof, df_den))
            elif test == "AD":
                statistic = anderson_statistic(
                    sim_vals, scipy.stats.f, params=(ndof, df_den)
                )

        ndof_sim.append(ndof)
        statistic_sim.append(statistic)

    ndof_sim = np.array(ndof_sim)
    statistic_sim = np.array(statistic_sim)

    # Compute p-value for test statistic evaluated on input values,
    # as fraction of simulations with test statistic greater than
    # data value
    p_cal = np.mean(statistic_sim > statistic_data)

    if return_ndof:
        return p_cal, statistic_data, statistic_sim, ndof_sim
    else:
        return p_cal, statistic_data, statistic_sim


def compute_MC_calibrated_LOBO_distribution_test(
    vals_block,
    vals_other,
    test="KS",
    dist="chi2",
    seed=0,
    n_for_F=998,
    p_for_F=16,
    verbose=False,
    n_mc_sims=1000,
):
    """Compute Monte-Carlo-calibrated leave-one-block-out distribution test.

    This routine takes separate "block" and "other" values to test. The
    desired distribution is fit to the "other" values, and then the test
    statistic is used to compare the "block" values to this distribution.
    The test statistic is calibrated by generating many Monte Carlo
    realizations of the "block" and "other" values, re-fitting the
    distribution to the "other" values, and re-computing the test
    statistic for the "block" values.

    Parameters
    ----------
    vals_block, vals_other : np.ndarray
        Arrays of values (see docstring for explanation)).
    test : str, optional
        Test to apply. Must be one of "KS" (Kolmogoriv-Smirnov)
        or "AD" (Anderson-Darling). Default: "KS".
    dist : str, optional
        Distribution to test. Must be one of "chi2" or "F".
        Default: "chi2".
    seed : int, optional
        Seed for random number generator. Default: 0.
    n_for_F, p_for_F: float, optional
        n and p parameters for "df2" of the F distribution.
        The p parameter for "df1" will be fit separately from
        `p_for_F`. Defaults: 998, 16.
    verbose : bool, optional
        Use `tqdm` package to display progress bar. Default: False.
    n_mc_sims : int, optional
        Number of Monte Carlo realizations of distribution.
        Default: 1000.

    Returns
    -------
    p_cal : float
        Calibrated p-value corresponding to chosen test.
    statistic_data : float
        Test statistic evaluated on input values.
    statistic_sim : np.ndarray
        Array of test statistics computed via Monte Carlo.
    """

    _TESTS = ["KS", "AD"]
    _DISTS = ["chi2", "F"]

    # Ensure that valid test statistic and distribution are being requested
    if test not in _TESTS:
        raise NotImplementedError(
            f"Test {test} not implemented (must be one of {_TESTS})"
        )

    if dist not in _DISTS:
        raise NotImplementedError(
            f"Distribution {dist} not implemented (must be one of {_DISTS})"
        )

    # Fit specified distribution to "other" input values, and compute desired
    # test statistic using "block" input values
    if dist == "chi2":
        ndof_data = fit_chi2_to_array(vals_other)

        if test == "KS":
            statistic_data, _ = scipy.stats.kstest(
                vals_block, "chi2", args=(ndof_data,)
            )
        elif test == "AD":
            statistic_data = anderson_statistic(
                vals_block, scipy.stats.chi2, params=(ndof_data,)
            )

    elif dist == "F":
        ndof_data, _ = fit_F_to_scaled_array(vals_other, n_for_F, p_for_F)

        # Set second parameter for F distribution
        df_den = n_for_F - p_for_F + 1

        # Note that input values need to be scaled before comparing to
        # F distribution
        if test == "KS":
            statistic_data, _ = scipy.stats.kstest(
                vals_block * F_scaling(n_for_F, p_for_F, ndof_data),
                "f",
                args=(ndof_data, df_den),
            )
        elif test == "AD":
            statistic_data = anderson_statistic(
                vals_block * F_scaling(n_for_F, p_for_F, ndof_data),
                scipy.stats.f,
                params=(ndof_data, df_den),
            )

    # Define lists to store fit parameters and test statistics for
    # each Monte Carlo simulation
    ndof_sim, statistic_sim = [], []

    # Initialize random number generator
    rng = np.random.default_rng(seed=seed)

    # Get sizes of datasets
    n_mocks_block = len(vals_block)
    n_mocks_other = len(vals_other)
    n_mocks_total = n_mocks_block + n_mocks_other

    # Use tqdm to print status during loop over Monte Carlo simulations,
    # if desired
    if verbose:
        s_range = trange(n_mc_sims)
    else:
        s_range = range(n_mc_sims)

    # Loop over MC sims
    for _ in s_range:
        if dist == "chi2":
            # Generate n_mocks random draws from chi^2 distribution,
            # and fit n_dof
            sim_vals_total = rng.chisquare(df=ndof_data, size=n_mocks_total)
            sim_vals_block = sim_vals_total[:n_mocks_block]
            sim_vals_other = sim_vals_total[n_mocks_block:]

            ndof = fit_chi2_to_array(sim_vals_other)

            if test == "KS":
                statistic, _ = scipy.stats.kstest(sim_vals_block, "chi2", args=(ndof,))
            elif test == "AD":
                statistic = anderson_statistic(
                    sim_vals_block, scipy.stats.chi2, params=(ndof,)
                )

        elif dist == "F":
            # Generate n_mocks random draws from F distribution,
            # and fit n_dof
            sim_vals_total = rng.f(dfnum=ndof_data, dfden=df_den, size=n_mocks_total)
            sim_vals_block = sim_vals_total[:n_mocks_block]
            sim_vals_other = sim_vals_total[n_mocks_block:]

            ndof = scipy.stats.f.fit(sim_vals_other, fdfd=df_den, floc=0, fscale=1)[0]

            if test == "KS":
                statistic, _ = scipy.stats.kstest(
                    sim_vals_block, "f", args=(ndof, df_den)
                )
            elif test == "AD":
                statistic = anderson_statistic(
                    sim_vals_block, scipy.stats.f, params=(ndof, df_den)
                )

        ndof_sim.append(ndof)
        statistic_sim.append(statistic)

    ndof_sim = np.array(ndof_sim)
    statistic_sim = np.array(statistic_sim)

    # Compute p-value for test statistic evaluated on input values,
    # as fraction of simulations with test statistic greater than
    # data value
    p_cal = np.mean(statistic_sim > statistic_data)

    return p_cal, statistic_data, statistic_sim


def find_symmetric_roots(
    func,
    root,
    x0,
    x_lo_bound,
    x_hi_bound,
):
    """Find roots of a function on either side of a certain point.

    Parameters
    ----------
    func : callable
        Scalar function to use for root-finding.
    root : float
        Value of desired root.
    x0 : float
        Point dividing two search regions for roots.
    x_lo_bound, x_hi_bound : float
        Lower and upper bounds of search regions for roots.

    Returns
    -------
    root_lo, root_high : float
        Roots on either side of `x0`.
    """

    resd_lo = scipy.optimize.minimize_scalar(
        lambda x: np.abs(func(x) - root), bounds=[x_lo_bound, x0]
    )
    resd_hi = scipy.optimize.minimize_scalar(
        lambda x: np.abs(func(x) - root), bounds=[x0, x_hi_bound]
    )

    if not resd_lo.success or not resd_hi.success:
        raise RuntimeError("One of the root-finders failed")

    return resd_lo.x, resd_hi.x
