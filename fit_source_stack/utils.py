import numpy as np

from . import containers


def covariance(a, corr=False):

    am = a - np.mean(a, axis=0)

    cov = np.sum(am[:, np.newaxis, :] * am[:, :, np.newaxis], axis=0) / float(am.shape[0] - 1)

    if corr:
        diag = np.diag(cov)
        cov = cov * tools.invert_no_zero(np.sqrt(diag[np.newaxis, :] * diag[:, np.newaxis]))

    return cov


def combine_pol(stack):

    y = stack['stack'][:]
    w = stack['weight'][:]

    ax = list(stack['stack'].attrs['axis']).index('pol')

    wz = np.sum(w, axis=ax)

    z = np.sum(w * y, axis=ax) * tools.invert_no_zero(wz)

    return z, wz


def initialize_pol(cnt):

    nfreq = cnt.freq.size
    mpol = cnt.pol.size

    stack = np.zeros((mpol+1, nfreq), dtype=cnt.stack.dtype)
    stack[0:mpol] = cnt.stack[:]

    temp, _ = combine_pol(cnt)
    stack[-1] = temp

    return stack


def load_mocks(mocks):

    if isinstance(mocks, str) or isinstance(mocks, containers.MockFrequencyStackByPol):

        if isinstance(mocks, str):
            mocks = containers.MockFrequencyStackByPol.from_file(mocks)

        pol = np.append(mocks.pol[:], 'I')
        npol = pol.size
        mpol = npol - 1

        # Create output container
        out = containers.MockFrequencyStackByPol(pol=pol, axes_from=mocks)

        # Load the mocks into an array
        mock_stack = out['stack'][:]
        mock_weight = out['weight'][:]

        mock_stack[:, 0:mpol, :] = mocks['stack'][:]
        mock_weight[:, 0:mpol, :] = mocks['weight'][:]

        ms, mw = combine_pol(mocks)
        mock_stack[:, -1, :] = ms
        mock_weight[:, -1, :] = mw

    else:

        if isinstance(mocks[0], str):
            mocks = [containers.FrequencyStackByPol.from_file(mck) for mck in mocks]

        nmocks = len(mocks)

        mock0 = mocks[0]
        freq = mock0.index_map['freq']

        pol = np.append(mock0.pol[:], 'I')
        npol = pol.size
        mpol = npol - 1

        # Create output container
        out = containers.MockFrequencyStackByPol(freq=freq, pol=pol, mock=np.arange(nmocks, dtype=np.int))

        # Load the mocks into an array
        mock_stack = out['stack'][:]
        mock_weight = out['weight'][:]

        for mm, mck in enumerate(mocks):
            mock_stack[mm, 0:mpol, :] = mck['stack'][:]
            mock_weight[mm, 0:mpol, :] = mck['weight'][:]

            ms, mw = combine_pol(mck)
            mock_stack[mm, -1, :] = ms
            mock_weight[mm, -1, :] = mw

    return out