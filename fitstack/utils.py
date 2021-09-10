"""Utililites to prepare the data for the fit."""
import os
import glob

import h5py

import numpy as np
from scipy.fftpack import next_fast_len

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


def unravel_covariance(cov, npol, nfreq):
    """Separate the covariance matrix into sub-arrays based on polarisation.

    Parameters
    ----------
    cov : np.ndarray[npol * nfreq, npol * nfreq]
        Covariance matrix.
    npol : int
        Number of polarisations.
    nfreq : int
        Number of frequencies.

    Returns
    -------
    cov_by_pol : np.ndarray[npol, npol, nfreq, nfreq]
        Covariance matrix reformatted such that cov_by_pol[i,j]
        gives the covariance between polarisation i and j as
        a function of frequency offset.
    """

    cov_by_pol = np.zeros((npol, npol, nfreq, nfreq), dtype=cov.dtype)

    for aa in range(npol):

        slc_aa = slice(aa * nfreq, (aa + 1) * nfreq)

        for bb in range(npol):

            slc_bb = slice(bb * nfreq, (bb + 1) * nfreq)

            cov_by_pol[aa, bb] = cov[slc_aa, slc_bb]

    return cov_by_pol


def ravel_covariance(cov_by_pol):
    """Collapse the covariance matrix over the polarisation axes.

    Parameters
    ----------
    cov_by_pol : np.ndarray[npol, npol, nfreq, nfreq]
        Covariance matrix as formatted by the unravel_covariance method.

    Returns
    -------
    cov : np.ndarray[npol * nfreq, npol * nfreq]
        The covariance matrix flattened into the format required for
        inversion and subsequent likelihood computation.
    """

    npol, _, nfreq, _ = cov_by_pol.shape
    ntot = npol * nfreq

    cov = np.zeros((ntot, ntot), dtype=cov_by_pol.dtype)

    for aa in range(npol):

        slc_aa = slice(aa * nfreq, (aa + 1) * nfreq)

        for bb in range(npol):

            slc_bb = slice(bb * nfreq, (bb + 1) * nfreq)

            cov[slc_aa, slc_bb] = cov_by_pol[aa, bb]

    return cov


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def shift_and_convolve(freq, template, offset=0.0, kernel=None):
    """Shift a template and (optionally) convolve with a kernel.

    Parameters
    ----------
    freq : np.ndarray[nfreq,]
        Frequency offset in MHz.
    template : np.ndarray[..., nfreq]
        Template for the signal.
    offset : float
        Central frequency offset in MHz.
    kernel : np.ndarray[..., nfreq]
        Kernel to convolve with the template.

    Returns
    -------
    template_sc : np.ndarray[..., nfreq]
        Template after shifting by offset and convolving with kernel.
    """

    # Determine the size of the fft needed for convolution
    nfreq = freq.size
    assert nfreq == template.shape[-1]

    size = nfreq if kernel is None else nfreq + kernel.shape[-1] - 1
    fsize = next_fast_len(int(size))
    fslice = slice(0, int(size))

    # Determine the delay corresponding to the frequency offset
    df = np.abs(freq[1] - freq[0]) * 1e6
    tau = np.fft.rfftfreq(fsize, d=df) * 1e6

    shift = np.exp(-2.0j * np.pi * tau * offset)

    # Take the fft and apply the delay
    fft_model = np.fft.rfft(template, fsize, axis=-1) * shift

    # Multiply by the fft of the kernel (if provided)
    if kernel is not None:
        fft_model *= np.fft.rfft(kernel, fsize, axis=-1)

    # Perform the inverse fft and center appropriately
    model = np.fft.irfft(fft_model, fsize, axis=-1)[..., fslice].real
    model = _centered(model, template.shape)

    return model


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


def initialize_pol(cnt, pol=None, combine=False):
    """Select the stack data for the desired polarisations.

    Parameters
    ----------
    cnt : FrequencyStackByPol
        The source stack.
    pol : list of str
        The polarisations to select.  If not provided,
        then ["XX", "YY"] is assumed.
    combine : bool
        Add an element to the polarisation axis that is
        the weighted sum of XX and YY.

    Returns
    -------
    stack : np.ndarray[..., npol, nfreq]
        The stack dataset for the selected polarisations.
        If combine is True, there will be an additional
        element that is the weighted sum of the stack for
        the "XX" and "YY" polarisations.
    weight : np.ndarray[..., npol, nfreq]
        The weight dataset for the selected polarisations.
        If combine is True, there will be an additional
        element that is the sum of the weights for
        the "XX" and "YY" polarisations.
    """

    if pol is None:
        pol = ["XX", "YY"]

    cpol = list(cnt.pol)
    ipol = np.array([cpol.index(pstr) for pstr in pol])

    num_freq = cnt.freq.size
    num_cpol = ipol.size

    num_pol = num_cpol + int(combine)

    ax = list(cnt.stack.attrs["axis"]).index("pol")
    shp = list(cnt.stack.shape)
    shp[ax] = num_pol

    stack = np.zeros(shp, dtype=cnt.stack.dtype)
    weight = np.zeros(shp, dtype=cnt.stack.dtype)

    slc_in = (slice(None),) * ax + (ipol,)
    slc_out = (slice(None),) * ax + (slice(0, num_cpol),)

    stack[slc_out] = cnt["stack"][slc_in]
    weight[slc_out] = cnt["weight"][slc_in]

    if combine:
        slc_out = (slice(None),) * ax + (-1,)
        temp, wtemp = combine_pol(cnt)
        stack[slc_out] = temp
        weight[slc_out] = wtemp
        cpol.append("I")

    return stack, weight, cpol


def load_mocks(mocks, pol=None):
    """Load the mock catalog stacks.

    Parameters
    ----------
    mocks : list of str, FrequencyStackByPol, or MockFrequencyStackByPol; or glob
        Set of stacks on mock catalogs.  This can either be a
        MockFrequencyStackByPol container or a list of
        FrequencyStackByPol or MockFrequencyStackByPol containers.
        It can also be a filename or list of filenames that
        hold these types of containers and will be loaded from disk.
    pol : list of str
        Desired polarisations.  Defaults to ["XX", "YY"].

    Returns
    -------
    out : MockFrequencyStackByPol
        All mock catalogs in a single container.
    """

    if pol is None:
        pol = ["XX", "YY"]

    pol = np.atleast_1d(pol)

    if isinstance(mocks, containers.MockFrequencyStackByPol):

        if not np.array_equal(mocks.pol, pol):
            raise RuntimeError(
                "The mock catalogs that were provided have the incorrect polarisations."
            )

        out = mocks

    else:

        if isinstance(mocks, str):
            mocks = sorted(glob.glob(mocks))

        temp = []
        for mfile in mocks:
            if isinstance(mfile, str):
                pol_sel = determine_pol_sel(mfile, pol=pol)
                temp.append(
                    containers.MockFrequencyStackByPol.from_file(mfile, pol_sel=pol_sel)
                )
            else:
                if not np.array_equal(mfile.pol, pol):
                    raise RuntimeError(
                        "The mock catalogs that were provided have the incorrect polarisations."
                    )
                temp.append(mfile)

        nmocks = [
            mock.index_map["mock"].size if "mock" in mock.index_map else 1
            for mock in temp
        ]

        boundaries = np.concatenate(([0], np.cumsum(nmocks)))

        out = containers.MockFrequencyStackByPol(
            mock=int(boundaries[-1]), axes_from=temp[0]
        )

        for mm, (mock, nm) in enumerate(zip(temp, nmocks)):

            if nm > 1:
                slc_out = slice(boundaries[mm], boundaries[mm + 1])
            else:
                slc_out = boundaries[mm]

            out.stack[slc_out] = mock.stack[:]
            out.weight[slc_out] = mock.weight[:]

    return out


def determine_pol_sel(filename, pol=None):
    """Find indices into the pol axis of a file that yield the desired polarisations.

    Parameters
    ----------
    filename : str
        Name of the file.
    pol : list of str
        Desired polarisations.  Defaults to ["XX", "YY"].

    Returns
    -------
    ipol : np.ndarray[npol,]
        Array of indices into the polarisation axis that yields
        the requested polarisations.
    """

    if pol is None:
        pol = ["XX", "YY"]

    pol = np.atleast_1d(pol)

    with h5py.File(filename, "r") as handler:
        fpol = list(handler["index_map"]["pol"][:].astype(str))

    ipol = np.array([fpol.index(pstr) for pstr in pol])

    return ipol


def find_file(search):
    """Find the most recent file matching a glob string.

    Parameters
    ----------
    search : str
        Glob string to search.

    Returns
    -------
    filename : str
        Most recently modified file that matches the search.
    """

    files = glob.glob(search)
    files.sort(reverse=True, key=os.path.getmtime)

    nfiles = len(files)

    if nfiles == 0:
        raise ValueError(f"Could not find file {search}")

    elif nfiles > 1:
        ostr = "\n".join([f"({ii+1}) {ff}" for ii, ff in enumerate(files)])
        logger.warning(
            f"Found {nfiles} files that match search criteria.  " "Using (1):\n" + ostr
        )

    return files[0]
