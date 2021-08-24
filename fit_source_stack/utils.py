"""Utililites to prepare the data for the fit."""

import numpy as np

from draco.util import tools

from . import containers


def covariance(a, corr=False):
    """Calculate the sample covariance over mock catalogs.

    Parameters
    ----------
    a : np.ndarray[nmock, nfreq, ...]

    corr : bool
        Return the correlation matrix instead of the covariance matrix.
        Default is False.

    Returns
    -------
    cov : np.ndarray[nfreq, nfreq,  ...]
        The sample covariance matrix (or correlation matrix).
    """

    am = a - np.mean(a, axis=0)

    cov = np.sum(am[:, np.newaxis, :] * am[:, :, np.newaxis], axis=0) / float(
        am.shape[0] - 1
    )

    if corr:
        diag = np.diag(cov)
        cov = cov * tools.invert_no_zero(
            np.sqrt(diag[np.newaxis, :] * diag[:, np.newaxis])
        )

    return cov


def combine_pol(stack):
    """Perform a weighted sum of the XX and YY polarisations.

    Parameters
    ----------
    stack : FrequencyStackByPol, MockFrequencyStackByPol
        The source stack.

    Returns
    -------
    z : np.ndarray
        The weighted sum of the stack dataset for the
        XX and YY polarisations.
    wz : np.ndarray
        The sum of the weight dataset for the
        XX and YY polarisations.
    """

    y = stack["stack"][:]
    w = stack["weight"][:]

    ax = list(stack["stack"].attrs["axis"]).index("pol")
    pol = list(stack.pol)

    flag = np.zeros_like(w)
    for pstr in ["XX", "YY"]:
        pp = pol.index(pstr)
        slc = (slice(None),) * ax + (pp,)
        flag[slc] = 1.0

    w = flag * w

    wz = np.sum(w, axis=ax)

    z = np.sum(w * y, axis=ax) * tools.invert_no_zero(wz)

    return z, wz


def initialize_pol(cnt):
    """Addn element to pol axis that is the weighted sum of XX and YY (labeled I).

    Parameters
    ----------
    cnt : FrequencyStackByPol
        The source stack.

    Returns
    -------
    stack : np.ndarray[npol+1, nfreq]
        The stack dataset with an additional element that is the weighted sum
        of the stack for the  "XX" and "YY" polarisations.
    weight : np.ndarray[npol+1, nfreq]
        The weight dataset with an additional element that is the sum of the weights
        for the "XX" and "YY" polarisations.
    """

    nfreq = cnt.freq.size
    mpol = cnt.pol.size

    stack = np.zeros((mpol + 1, nfreq), dtype=cnt.stack.dtype)
    weight = np.zeros((mpol + 1, nfreq), dtype=cnt.stack.dtype)

    stack[0:mpol] = cnt["stack"][:]
    weight[0:mpol] = cnt["weight"][:]

    temp, wtemp = combine_pol(cnt)
    stack[-1] = temp
    weight[-1] = wtemp

    return stack, weight


def load_mocks(mocks, pol_sel=None):
    """Load the mock catalog stacks.

    Parameters
    ----------
    mocks : MockFrequencyStackByPol, str, list of FrequencyStackByPol, list of str
        Set of stacks on mock catalogs.  This can either be a
        MockFrequencyStackByPol container or a list of FrequencyStackByPol
        containers.  It can also be a filename or list of filenames that
        hold these types of containers and will be loaded from disk.

    pol_sel : slice
        Load a subset of polarisations.  Only used if the mocks parameter
        contains filename(s).

    Returns
    -------
    out : MockFrequencyStackByPol
        All mock catalogs in a common container with an
        intensity polarisation added to the pol axis.
    """

    if isinstance(mocks, str) or isinstance(mocks, containers.MockFrequencyStackByPol):

        if isinstance(mocks, str):
            mocks = containers.MockFrequencyStackByPol.from_file(
                mocks, pol_sel=pol_sel, detect_subclass=False
            )

        pol = np.append(mocks.pol[:], "I")
        npol = pol.size
        mpol = npol - 1

        # Create output container
        out = containers.MockFrequencyStackByPol(pol=pol, axes_from=mocks)

        # Load the mocks into an array
        mock_stack = out["stack"][:]
        mock_weight = out["weight"][:]

        mock_stack[:, 0:mpol, :] = mocks["stack"][:]
        mock_weight[:, 0:mpol, :] = mocks["weight"][:]

        ms, mw = combine_pol(mocks)
        mock_stack[:, -1, :] = ms
        mock_weight[:, -1, :] = mw

    else:

        if isinstance(mocks[0], str):
            mocks = [
                containers.FrequencyStackByPol.from_file(mck, pol_sel=pol_sel)
                for mck in mocks
            ]

        nmocks = len(mocks)

        mock0 = mocks[0]
        freq = mock0.index_map["freq"]

        pol = np.append(mock0.pol[:], "I")
        npol = pol.size
        mpol = npol - 1

        # Create output container
        out = containers.MockFrequencyStackByPol(
            freq=freq, pol=pol, mock=np.arange(nmocks, dtype=np.int)
        )

        # Load the mocks into an array
        mock_stack = out["stack"][:]
        mock_weight = out["weight"][:]

        for mm, mck in enumerate(mocks):
            mock_stack[mm, 0:mpol, :] = mck["stack"][:]
            mock_weight[mm, 0:mpol, :] = mck["weight"][:]

            ms, mw = combine_pol(mck)
            mock_stack[mm, -1, :] = ms
            mock_weight[mm, -1, :] = mw

    return out
