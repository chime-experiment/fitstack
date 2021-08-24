"""Define models that can be fit to the source stack."""

import numpy as np
from scipy.fftpack import next_fast_len


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


class Model(object):
    """Baseclass for emcee models.

    Attributes
    ----------
    param_name : list of str
        Names of the model parameters.
    boundary : dict, {param_name: [lower_bound, upper_bound], ...}
        Default boundaries on the parameter values.
    """

    param_name = []
    boundary = {}

    def set_data(self, **kwargs):
        """Save any ancillary data needed to evaluate the probabily distribution.

        Parameters
        ----------
        kwargs : {attr: value, ...}
            Each keyword argument will be saved as an attribute
            that can be accessed by the methods that evaluate the
            log of the probability of observing the data given the
            model parameters.
        """

        for key, val in kwargs.items():
            setattr(self, key, val)

    def log_prior(self, theta):
        """Evaluate the log of the prior distribution.

        Parameters
        ----------
        theta : list
            Parameter values.

        Returns
        -------
        prob : float
            Logarithm of the prior distribution.  This will be 0 if the
            parameters are within the ranges specified in the boundary
            class attribute and -Inf if they are outside the ranges.
        """

        theta = np.array(theta)

        bounds = np.array([self.boundary[key] for key in self.param_name])

        if np.all((theta > bounds[:, 0]) & (theta < bounds[:, 1])):
            return 0.0
        else:
            return -np.inf

    def log_likelihood(self, theta):
        """Evaluate the log of the likelihood.

        Parameters
        ----------
        theta : list
            Parameter values.

        Returns
        -------
        logL : float
            Logarithm of the likelihood function.
        """

        mdl = self.model(theta)

        residual = self.data - mdl

        return -0.5 * np.matmul(residual.T, np.matmul(self.inv_cov, residual))

    def log_probability(self, theta):
        """Evaluate log of the probability of observing the data given the parameters.

        Parameters
        ----------
        theta : list
            Parameter values.

        Returns
        -------
        prob : float
            The product of the likelihood function and the prior distribution
            for the parameter values provided.
        """

        lp = self.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + self.log_likelihood(theta)


class SST(Model):
    """Scaled and shifted template model."""

    _name = "scaled_shifted_template"

    param_name = ["amp", "offset"]

    boundary = {"amp": [0.0, 4.0], "offset": [-1.0, 1.0]}

    def model(self, theta, freq=None, transfer=None, template=None):
        """Evaluate the model consisting of a scaled and shifted template.

        .. math::

            S(\Delta \nu) = A H(\Delta \nu) \circledast T(\Delta \nu - \nu_{0})

        where `S` is the stacked signal as a function of
        frequency offset :math:`\Delta \nu`, `A` is the amplitude,
        :math:`\nu_{0}` is the central frequency offset,
        and `H` is the transfer function.

        Parameters
        ----------
        theta : [amp, offset]
            Two element list containing the amplitude and offset parameters.
            The template will be multiplied by the amplitude and then shifted by
            the offset in MHz.
        freq : np.ndarray[nfreq]
            Frequency offset in MHz.  If this was not provided, then will
            default to the `freq` attribute.
        transfer : np.ndarray[..., nfreq]
            Transfer function of the pipeline.  The template is convolved
            with the transfer function.  If this was not provided,
            then will default to the `transfer` attribute.
        template : np.ndarray[..., nfreq]
            Template for the signal.  If this was not provided, then
            will default to the `template` attribute.

        Returns
        -------
        model : np.ndarray[..., nfreq]
            Model for the signal, which is the template convolved with the
            transfer function, shifted by frequency offset, and scaled by
            the amplitude.
        """

        if freq is None:
            freq = self.freq

        if transfer is None:
            transfer = self.transfer

        if template is None:
            template = self.template

        amp, offset = theta

        size = freq.size + transfer.size - 1

        fsize = next_fast_len(int(size))  # , True)
        fslice = slice(0, int(size))

        # Evaluate the model
        model = amp * template

        # Determine the shift
        df = np.abs(freq[1] - freq[0]) * 1e6
        tau = np.fft.rfftfreq(fsize, d=df) * 1e6

        shift = np.exp(-2.0j * np.pi * tau * offset)

        # Apply the transfer function and shift to the model
        tmodel = np.fft.irfft(
            np.fft.rfft(model, fsize) * np.fft.rfft(transfer, fsize) * shift, fsize
        )[fslice].real

        return _centered(tmodel, freq.size)


class Exponential(Model):
    """Exponential model."""

    _name = "exponential"

    param_name = ["amp", "scale", "offset"]

    boundary = {"amp": [0.0, 500.0], "scale": [0.1, 10.0], "offset": [-1.0, 1.0]}

    def model(self, theta, freq=None, transfer=None):
        """Evaluate the exponential model.

        .. math::

            S(\Delta \nu) = A H(\Delta \nu) \circledast exp(-|\Delta \nu - \nu{0}| / s)

        where `S` is the stacked signal as a function of
        frequency offset :math:`\Delta \nu`, `A` is the amplitude,
        `s` is the scale, :math:`\nu_{0}` is the central frequency offset,
        and `H` is the transfer function.

        Parameters
        ----------
        theta : [amp, scale, offset]
            Three element list containing the model parameters, which are
            the amplitude, scale (in MHz), and offset (in MHz).
        freq : np.ndarray[nfreq]
            Frequency offset in MHz.  If this was not provided, then will
            default to the `freq` attribute.
        transfer : np.ndarray[..., nfreq]
            Transfer function of the pipeline.  The exponential is convolved
            with the transfer function.  If this was not provided,
            then will default to the `transfer` attribute.

        Returns
        -------
        model : np.ndarray[..., nfreq]
            Model for the signal.
        """
        if freq is None:
            freq = self.freq

        if transfer is None:
            transfer = self.transfer

        amp, scale, offset = theta

        size = freq.size + transfer.size - 1

        fsize = next_fast_len(int(size))  # , True)
        fslice = slice(0, int(size))

        # Evaluate the model
        model = amp * np.exp(-np.abs(freq) / scale)

        # Determine the shift
        df = np.abs(freq[1] - freq[0]) * 1e6
        tau = np.fft.rfftfreq(fsize, d=df) * 1e6

        shift = np.exp(-2.0j * np.pi * tau * offset)

        # Apply the transfer function and shift to the model
        tmodel = np.fft.irfft(
            np.fft.rfft(model, fsize) * np.fft.rfft(transfer, fsize) * shift, fsize
        )[fslice].real

        return _centered(tmodel, freq.size)
