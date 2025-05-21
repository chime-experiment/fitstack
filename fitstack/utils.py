"""Utililites to prepare the data for the fit."""

import logging
import os
import glob
from pathlib import Path

import h5py
import numpy as np
from scipy.fftpack import next_fast_len

from caput import misc
from draco.util import tools

from . import containers

logger = logging.getLogger(__name__)


def covariance(a, corr=False):
    """Calculate the sample covariance over mock catalogs or power spectra.

    Parameters
    ----------
    a : np.ndarray[nmock, nx, ...]
        Array of mock data.
    corr : bool
        Return the correlation matrix instead of the covariance matrix.
        Default is False.

    Returns
    -------
    cov : np.ndarray[nx, nx,  ...]
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


def unravel_covariance(cov, npol, nx):
    """Separate the covariance matrix into sub-arrays based on polarisation.

    Parameters
    ----------
    cov : np.ndarray[npol * nx, npol * nx]
        Covariance matrix.
    npol : int
        Number of polarisations.
    nx : int
        Number of frequencies or k bins.

    Returns
    -------
    cov_by_pol : np.ndarray[npol, npol, nx, nx]
        Covariance matrix reformatted such that cov_by_pol[i,j]
        gives the covariance between polarisation i and j as
        a function of frequency offset or k.
    """

    cov_by_pol = np.zeros((npol, npol, nx, nx), dtype=cov.dtype)

    for aa in range(npol):

        slc_aa = slice(aa * nx, (aa + 1) * nx)

        for bb in range(npol):

            slc_bb = slice(bb * nx, (bb + 1) * nx)

            cov_by_pol[aa, bb] = cov[slc_aa, slc_bb]

    return cov_by_pol


def ravel_covariance(cov_by_pol):
    """Collapse the covariance matrix over the polarisation axes.

    Parameters
    ----------
    cov_by_pol : np.ndarray[npol, npol, nx, nx]
        Covariance matrix as formatted by the unravel_covariance method.

    Returns
    -------
    cov : np.ndarray[npol * nx, npol * nx]
        The covariance matrix flattened into the format required for
        inversion and subsequent likelihood computation.
    """

    npol, _, nx, _ = cov_by_pol.shape
    ntot = npol * nx

    cov = np.zeros((ntot, ntot), dtype=cov_by_pol.dtype)

    for aa in range(npol):

        slc_aa = slice(aa * nx, (aa + 1) * nx)

        for bb in range(npol):

            slc_bb = slice(bb * nx, (bb + 1) * nx)

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
    """Shift a stacking template and (optionally) convolve with a kernel.

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


def combine_pol(cnt):
    """Perform a weighted sum of the XX and YY polarisations.

    Parameters
    ----------
    cnt : container
        Input container. Can be one of FrequencyStackByPol,
        MockFrequencyStackByPol, PowerSpectrum1D, MockPowerSpectrum1D,
        PowerSpectrum2D, MockPowerSpectrum2D.

    Returns
    -------
    z : np.ndarray
        The weighted sum of the relevant dataset for the XX and YY polarisations.
    wz : np.ndarray
        The sum of the weights for the XX and YY polarisations.
    x : np.ndarray
        The weighted average of the independent coordinate (frequency lag, k,
        or dict with kpara and kperp keys) for the XX and YY polarisations.
    """

    _dset_name = {"stack": "stack", "ps2D": "spectrum", "ps1D": "spectrum"}

    if isinstance(cnt, containers.FrequencyStackByPol):
        data_type = "stack"
        copol_names = ["XX", "YY"]
        is_mock_cont = isinstance(cnt, containers.MockFrequencyStackByPol)
    else:
        copol_names = ["XX-XX", "YY-YY"]
        if isinstance(cnt, containers.PowerSpectrum2D):
            data_type = "ps2D"
            is_mock_cont = isinstance(cnt, containers.MockPowerSpectrum2D)
        else:
            data_type = "ps1D"
            is_mock_cont = isinstance(cnt, containers.MockPowerSpectrum1D)

    pol = list(cnt.index_map["pol"])

    # If operating on power spectra, check that ordering of k-bin centers is
    # identical for the two polarizations. (Otherwise, we shouldn't combine them.)
    if data_type == "ps1D":
        isort = np.argsort(cnt.k1D)
        ax = list(cnt.k1D.attrs["axis"]).index("pol")
        slc_XX = (slice(None),) * ax + (pol.index("XX-XX"),)
        slc_YY = (slice(None),) * ax + (pol.index("YY-YY"),)
        if not np.allclose(isort[slc_XX], isort[slc_YY]):
            raise RuntimeError(
                "Power spectrum k bins have different ordering "
                "for different polarizations, so can't combine"
            )

    y = cnt[_dset_name[data_type]]

    if (data_type == "stack") | (data_type == "ps2D"):
        w = cnt["weight"][:]
    else:
        w = tools.invert_no_zero(cnt["var"][:])

    ax = list(cnt[_dset_name[data_type]].attrs["axis"]).index("pol")

    flag = np.zeros_like(w)
    for pstr in copol_names:
        pp = pol.index(pstr)
        slc = (slice(None),) * ax + (pp,)
        flag[slc] = 1.0

    w = flag * w

    wz = np.sum(w, axis=ax)

    z = np.sum(w * y, axis=ax) * tools.invert_no_zero(wz)

    if data_type == "stack":
        # Frequencies are identical for XX and YY, so no average needed
        x = cnt.freq
    elif data_type == "ps2D":
        # k_par and k_perp are identical for XX and YY, so no average needed here either
        x = {"kpara": cnt.kpara, "kperp": cnt.kperp}
    else:
        x = cnt.k1D[:]
        if is_mock_cont:
            # Check that weights are identical for each mock
            if not np.allclose(w, w[0]):
                raise RuntimeError(
                    "Weights in MockPowerSpectrum1D are different for each mock."
                    "The current code implementation cannot handle this."
                )
            # Just use weights for first mock, modifying ax to account for
            # fact that we've selected a single element of the mock axis
            x = np.sum(w[0] * x, axis=ax-1) * tools.invert_no_zero(wz[0])
        else:
            x = np.sum(w * x, axis=ax) * tools.invert_no_zero(wz)

    return z, wz, x


def initialize_pol(cnt, pol=None, combine=False, return_signal_mask=False):
    """Select the data for the desired polarisations.

    Parameters
    ----------
    cnt : FrequencyStackByPol, PowerSpectrum1D, or PowerSpectrum2D
        Container with stack or power spectrum.
    pol : list of str
        The polarisations to select.  If not provided,
        then ["XX", "YY"] is assumed for stack or ["XX-XX", "YY-YY"]
        is assumed for power spectrum.
    combine : bool
        Add an element to the polarisation axis that is
        the weighted sum of XX and YY.
    return_signal_mask : bool
        Also return signal_mask for 2d power spectrum.
        Ignored if input container is not Powerspec2D.
        Default: False.

    Returns
    -------
    data : np.ndarray[..., npol, nx] or np.ndarray[..., npol, nkpara, nkperp]
        The stack or power spectrum dataset for the selected
        polarisations.
        If combine is True, there will be an additional
        element that is the weighted sum of the
        stack/power spectrum for
        the "XX" and "YY" polarisations.
    weight : np.ndarray[..., npol, nx] or np.ndarray[..., npol, nkpara, nkperp]
        The weight dataset for the selected polarisations.
        If combine is True, there will be an additional
        element that is the sum of the weights for
        the "XX" and "YY" polarisations.
    cpol : list of str
        List of polarizations in output arrays.
    x : np.ndarray[..., npol, nx] or dict
        Frequencies or k values for the selected polarizations.
        If dealing with 2d power spectrum, dict has kpara and kperp keys,
        each as np.ndarray[..., npol, nk].
    """

    _dset_name = {"stack": "stack", "ps2D": "spectrum", "ps1D": "spectrum"}

    if isinstance(cnt, containers.FrequencyStackByPol):
        data_type = "stack"
        if pol is None:
            pol = ["XX", "YY"]
    else:
        if pol is None:
            pol = ["XX-XX", "YY-YY"]

        if isinstance(cnt, containers.PowerSpectrum2D):
            data_type = "ps2D"
        else:
            data_type = "ps1D"

    cpol = list(cnt.index_map["pol"])
    ipol = np.array([cpol.index(pstr) for pstr in pol])

    num_cpol = ipol.size

    num_pol = num_cpol + int(combine)

    if isinstance(cnt, containers.FrequencyStackByPol):
        dset = "stack"
    else:
        dset = "spectrum"

    ax = list(cnt[dset].attrs["axis"]).index("pol")
    shp = list(cnt[dset].shape)
    shp[ax] = num_pol

    data = np.zeros(shp, dtype=cnt[dset].dtype)
    weight = np.zeros(shp, dtype=cnt[dset].dtype)
    if data_type != "ps2D":
        x = np.zeros(shp, dtype=cnt[dset].dtype)
    else:
        x = {
            "kpara": np.zeros(tuple(shp[:-2]) + (len(cnt.kpara),), dtype=cnt.kpara.dtype),
            "kperp": np.zeros(
                tuple(shp[:-2]) + (len(cnt.kperp),), dtype=cnt.kperp.dtype
            ),
        }

    slc_in = (slice(None),) * ax + (ipol,)
    slc_out = (slice(None),) * ax + (slice(0, num_cpol),)

    data[slc_out] = cnt[dset][slc_in]
    if (data_type == "stack") | (data_type == "ps2D"):
        weight[slc_out] = cnt["weight"][slc_in]
    else:
        weight[slc_out] = tools.invert_no_zero(cnt.datasets["var"][slc_in])

    if data_type == "stack":
        x[slc_out] = cnt.freq[..., :]
    elif data_type == "ps2D":
        x["kpara"][slc_out] = cnt.kpara[..., :]
        x["kperp"][slc_out] = cnt.kperp[..., :]
    else:
        x[slc_out] = cnt.k1D[:]

    if data_type == "ps2D" and return_signal_mask:
        signal_mask = np.zeros(shp, dtype=cnt.mask.dtype)
        signal_mask[slc_out] = cnt.mask[slc_in]

    if combine:
        old_slc_out = slc_out
        slc_out = (slice(None),) * ax + (-1,)
        temp, wtemp, xtemp = combine_pol(cnt)
        data[slc_out] = temp
        weight[slc_out] = wtemp
        if data_type == "ps2D":
            x["kpara"][slc_out] = xtemp["kpara"]
            x["kperp"][slc_out] = xtemp["kperp"]
            if return_signal_mask:
                signal_mask[slc_out] = np.all(signal_mask[old_slc_out], axis=ax)
        else:
            x[slc_out] = xtemp
        cpol.append("I")

    if return_signal_mask:
        return data, weight, cpol, x, signal_mask
    else:
        return data, weight, cpol, x


def average_data(cnt, pol=None, combine=True, sort=True):
    """Calculate the mean and variance of a set of stacks or power spectra.

    Parameters
    ----------
    cnt : MockFrequencyStackByPol, MockPowerSpectrum1D, or MockPowerSpectrum2D
        Container with stacks or power spectra to average.
    pol : list of str
        The polarisations to select.  If not provided,
        then ["XX", "YY"] is assumed for stacks or ["XX-XX", "YY-YY"]
        is assumed for power spectra.
    combine : bool
        Add an element to the polarisation axis that is
        the weighted sum of XX and YY.  Default is True.
    sort : bool
        Sort the frequency offset or k axis in ascending order.
        Ignored if working with 2d power spectra.
        Default is True.

    Returns
    -------
    avg : FrequencyStackByPol, PowerSpectrum1D, or PowerSpectrum2D
        Container that has collapsed over the mock axis.
        The stack or spectrum dataset contains the mean. For stacks,
        the weight dataset contains the inverse variance,
        while for power spectra, the var dataset contains
        the variance (1d) or the weight dataset contains the
        inverse variance (2d).
    """

    darr, _, dpol, dx = initialize_pol(cnt, pol=pol, combine=combine)
    ndata = darr.shape[0]

    # freq/k should always be real
    dx = np.real(dx)

    # If requested, sort by freq/k
    if sort and not isinstance(cnt, containers.MockPowerSpectrum2D):
        isort = np.argsort(dx, axis=-1)
        dx = np.take_along_axis(dx, isort, axis=-1)
        darr = np.take_along_axis(darr, isort, axis=-1)

    # Make new container with mean and variance over mock axis
    if isinstance(cnt, containers.MockFrequencyStackByPol):
        avg = containers.FrequencyStackByPol(
            pol=np.array(dpol), freq=cnt.freq, attrs_from=cnt
        )
        avg.stack[:] = np.mean(darr, axis=0)
        avg.weight[:] = tools.invert_no_zero(np.var(darr, axis=0))
    elif isinstance(cnt, containers.MockPowerSpectrum2D):
        avg = containers.PowerSpectrum2D(
            pol=np.array(dpol),
            delay=cnt.index_map["delay"],
            uv_dist=cnt.index_map["uv_dist"],
            attrs_from=cnt,
            distributed=False,
        )
        avg.spectrum[:] = np.mean(darr, axis=0)
        avg.kpara[:] = cnt.kpara[:]
        avg.kperp[:] = cnt.kperp[:]

        if darr.shape[0] == 1:
            # Set weights to unity for single mock
            avg.weight[:] = np.ones_like(avg.spectrum[:], dtype=int)
        else:
            # Variance calculation for multiple mocks
            avg.weight[:] = tools.invert_no_zero(np.var(darr, axis=0))
    else:
        avg = containers.PowerSpectrum1D(
            pol=np.array(dpol), k=cnt.index_map["k"], attrs_from=cnt, distributed=False
        )
        avg.k1D[:] = np.mean(dx, axis=0)
        avg.spectrum[:] = np.mean(darr, axis=0)
        avg.var[:] = np.var(darr, axis=0)

    avg.attrs["num"] = ndata

    return avg


def load_pol(filename, pol=None):
    """Load a file, down-selecting along the polarisation axis.

    This is a wrapper for the from_file method of
    container.BaseContainer that first opens the file
    using h5py to determines the appropriate container type
    and indices into the polarisation axis.

    Parameters
    ----------
    filename : str
        Name of the file.
    pol : list of str
        Desired polarisations.  Defaults to ["XX", "YY"] for stack
        or ["XX-XX", "YY-YY"] for power spectrum.

    Returns
    -------
    out : subclass of containers.BaseContainer
        File in the appropriate container with
        the requested polarisations.
    """

    with h5py.File(filename, "r") as handler:
        container_path = handler.attrs["__memh5_subclass"]
        fpol = list(handler["index_map"]["pol"][:].astype(str))

    if container_path in [
        "draco.core.containers.FrequencyStackByPol",
        "draco.core.containers.MockFrequencyStackByPol",
    ]:
        if pol is None:
            pol = ["XX", "YY"]
    elif container_path in [
        "draco.core.containers.PowerSpectrum2D", 
        "draco.core.containers.MockPowerSpectrum2D", 
        "draco.core.containers.PowerSpectrum1D",
        "draco.core.containers.MockPowerSpectrum1D",
    ]:
        if pol is None:
            pol = ["XX-XX", "YY-YY"]
    else:
        raise RuntimeError(
            f"Container type of file ({container_path}) not recognized"
        )

    pol = np.atleast_1d(pol)

    ipol = np.array([fpol.index(pstr) for pstr in pol])

    Container = misc.import_class(container_path)

    return Container.from_file(filename, pol_sel=ipol, distributed=False)


def load_mocks(mocks, pol=None):
    """Load the mock catalog stacks/noise power spectra.

    Parameters
    ----------
    mocks : list of str; container; list of containers; or glob
        Set of stacks on mock catalogs or noise power spectra.
        This can either be a MockFrequencyStackByPol,
        MockPowerSpectrum1D, or MockPowerSpectrum2D container; a list of
        FrequencyStackByPol, PowerSpectrum1D, or PowerSpectrum2D containers;
        or a filename or list of filenames that
        hold these types of containers and will be loaded from disk.
    pol : list of str
        Desired polarisations.  Defaults to ["XX", "YY"] for stacks or
        ["XX-XX", "YY-YY"] for power spectra.

    Returns
    -------
    out : MockFrequencyStackByPol, MockPowerSpectrum1D, or MockPowerSpectrum2D
        All mock catalogs or power spectra in a single container.
    """

    if isinstance(
        mocks,
        (
            containers.MockFrequencyStackByPol,
            containers.MockPowerSpectrum1D,
            containers.MockPowerSpectrum2D,
        ),
    ):
        if pol is None:
            if isinstance(mocks, containers.MockFrequencyStackByPol):
                pol = ["XX", "YY"]
            else:
                pol = ["XX-XX", "YY-YY"]

        pol = np.atleast_1d(pol)

        if not np.array_equal(mocks.pol, pol):
            raise RuntimeError(
                "The mock catalogs/power spectra that were provided have "
                "incorrect polarisations."
            )

        out = mocks

    else:

        if isinstance(mocks, str):
            mocks = sorted(glob.glob(mocks))

            if pol is None:
                with h5py.File(mocks[0], "r") as handler:
                    container_type = handler.attrs["__memh5_subclass"]
                    if container_type == "draco.core.containers.FrequencyStackByPol":
                        pol = ["XX", "YY"]
                    else:
                        pol = ["XX-XX", "YY-YY"]

        else:
            if pol is None:
                if isinstance(mocks[0], containers.MockFrequencyStackByPol):
                    pol = ["XX", "YY"]
                else:
                    pol = ["XX-XX", "YY-YY"]

        pol = np.atleast_1d(pol)

        temp = []
        for mfile in mocks:
            if isinstance(mfile, (str, Path)):
                temp.append(load_pol(mfile, pol=pol))
            else:
                if not np.array_equal(mfile.pol, pol):
                    raise RuntimeError(
                        "The mock catalogs/power spectra that were provided have "
                        "incorrect polarisations."
                    )
                temp.append(mfile)

        nmocks = [
            mock.index_map["mock"].size if "mock" in mock.index_map else 1
            for mock in temp
        ]

        boundaries = np.concatenate(([0], np.cumsum(nmocks)))

        if isinstance(temp[0], containers.FrequencyStackByPol):
            out = containers.MockFrequencyStackByPol(
                mock=np.arange(boundaries[-1], dtype=int),
                axes_from=temp[0],
                attrs_from=temp[0],
            )
        elif isinstance(temp[0], containers.PowerSpectrum2D):
            out = containers.MockPowerSpectrum2D(
                mock=np.arange(boundaries[-1], dtype=int),
                axes_from=temp[0],
                attrs_from=temp[0],
            )
        else:
            out = containers.MockPowerSpectrum1D(
                mock=np.arange(boundaries[-1], dtype=int),
                axes_from=temp[0],
                attrs_from=temp[0],
            )

        for mm, (mock, nm) in enumerate(zip(temp, nmocks)):

            if nm > 1:
                slc_out = slice(boundaries[mm], boundaries[mm + 1])
            else:
                slc_out = boundaries[mm]

            if isinstance(temp[0], containers.FrequencyStackByPol):
                out.stack[slc_out] = mock.stack[:]
                out.weight[slc_out] = mock.weight[:]
            elif isinstance(temp[0], containers.PowerSpectrum2D):
                out.spectrum[slc_out] = mock.spectrum[:]
                out.weight[slc_out] = mock.weight[:]
                if mm == 0:
                    out.mask[:] = mock.mask[:]
                    out.kpara[:] = mock.kpara[:]
                    out.kperp[:] = mock.kperp[:]
            else:
                out.spectrum[slc_out] = mock.spectrum[:]
                out.samp_var[slc_out] = mock.samp_var[:]
                if mm == 0:
                    out.var[:] = mock.var[:]

        if isinstance(temp[0], containers.PowerSpectrum1D):
            out.k1D[:] = mock.k1D[:]

    return out


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
