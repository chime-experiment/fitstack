import numpy as np
import scipy.fft

# MCMC FIT
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def log_prior(theta, param_name=None):

    theta = np.array(theta)

    bounds = np.array([BOUNDARY[key] for key in param_name])

    if np.all((theta > bounds[:, 0]) & (theta < bounds[:, 1])):
        return 0.0
    else:
        return -np.inf



class Model(object):
    
    BOUNDARY = {}
    
    def log_prior(self, theta, param_name=None):
        
        theta = np.array(theta)

        bounds = np.array([self.BOUNDARY[key] for key in param_name])

        if np.all((theta > bounds[:, 0]) & (theta < bounds[:, 1])):
            return 0.0
        else:
            return -np.inf


# scaled and shifted template fit
class SST(Model):

    BOUNDARY = {'amp':    [0.0, 500.0],
                'offset': [-1.0, 1.0]}

    def model(self, theta, freq=None, transfer=None, template=None):

        amp, offset = theta

        size = freq.size + transfer.size - 1

        fsize = scipy.fft.next_fast_len(int(size), True)
        fslice = slice(0, int(size))

        # Evaluate the model
        model = amp * template

        # Determine the shift
        df = np.abs(freq[1] - freq[0]) * 1e6
        tau = np.fft.rfftfreq(fsize, d=df) * 1e6

        shift = np.exp(-2.0J * np.pi * tau * offset)

        # Apply the transfer function and shift to the model
        tmodel = np.fft.irfft(np.fft.rfft(model, fsize) *
                              np.fft.rfft(transfer, fsize) *
                              shift, fsize)[fslice].real

        return _centered(tmodel, freq.size)

    def log_likelihood(self, theta, data, inv_cov, freq, transfer, template):

        mdl = self.model(theta, freq=freq, transfer=transfer, template=template)

        residual = data - mdl

        return -0.5 * np.matmul(residual.T, np.matmul(inv_cov, residual))

    def log_probability(self, theta, param_name, data, inv_cov, freq, transfer, template):

        lp = self.log_prior(theta, param_name=param_name)

        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + self.log_likelihood(theta, data, inv_cov, freq, transfer, template)


def sst_model(theta, freq=None, transfer=None, template=None):

    amp, offset = theta

    size = freq.size + transfer.size - 1

    fsize = scipy.fft.next_fast_len(int(size), True)
    fslice = slice(0, int(size))

    # Evaluate the model
    model = amp * template

    # Determine the shift
    df = np.abs(freq[1] - freq[0]) * 1e6
    tau = np.fft.rfftfreq(fsize, d=df) * 1e6

    shift = np.exp(-2.0J * np.pi * tau * offset)

    # Apply the transfer function and shift to the model
    tmodel = np.fft.irfft(np.fft.rfft(model, fsize) *
                          np.fft.rfft(transfer, fsize) *
                          shift, fsize)[fslice].real

    return _centered(tmodel, freq.size)


def sst_model_log_likelihood(theta, data, inv_cov, freq, transfer, template):

    mdl = sst_model(theta, freq=freq, transfer=transfer, template=template)

    residual = data - mdl

    return -0.5 * np.matmul(residual.T, np.matmul(inv_cov, residual))


def sst_model_log_probability(theta, param_name, data, inv_cov, freq, transfer, template):

    lp = log_prior(theta, param_name=param_name)

    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + sst_model_log_likelihood(theta, data, inv_cov, freq, transfer, template)



# exponential model
class Exponential(Model):
    
    BOUNDARY = {'amp':    [0.0, 500.0],
                'scale':  [0.1, 10.0],
                'offset': [-1.0, 1.0]}
    
    def model(self, theta, freq=None, transfer=None):
        
        amp, scale, offset = theta

        size = freq.size + transfer.size - 1

        fsize = scipy.fft.next_fast_len(int(size), True)
        fslice = slice(0, int(size))

        # Evaluate the model
        model = amp * np.exp(-np.abs(freq) / scale)

        # Determine the shift
        df = np.abs(freq[1] - freq[0]) * 1e6
        tau = np.fft.rfftfreq(fsize, d=df) * 1e6

        shift = np.exp(-2.0J * np.pi * tau * offset)

        # Apply the transfer function and shift to the model
        tmodel = np.fft.irfft(np.fft.rfft(model, fsize) *
                              np.fft.rfft(transfer, fsize) *
                              shift, fsize)[fslice].real

        return _centered(tmodel, freq.size)


    def log_likelihood(self, theta, data, inv_cov, freq, transfer):

        mdl = self.model(theta, freq=freq, transfer=transfer)

        residual = data - mdl

        return -0.5 * np.matmul(residual.T, np.matmul(inv_cov, residual))


    def log_probability(self, theta, param_name, data, inv_cov, freq, transfer):

        lp = self.log_prior(theta, param_name=param_name)

        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + self.log_likelihood(theta, data, inv_cov, freq, transfer)



def exp_model(theta, freq=None, transfer=None):

    amp, scale, offset = theta

    size = freq.size + transfer.size - 1

    fsize = scipy.fft.next_fast_len(int(size), True)
    fslice = slice(0, int(size))

    # Evaluate the model
    model = amp * np.exp(-np.abs(freq) / scale)

    # Determine the shift
    df = np.abs(freq[1] - freq[0]) * 1e6
    tau = np.fft.rfftfreq(fsize, d=df) * 1e6

    shift = np.exp(-2.0J * np.pi * tau * offset)

    # Apply the transfer function and shift to the model
    tmodel = np.fft.irfft(np.fft.rfft(model, fsize) *
                          np.fft.rfft(transfer, fsize) *
                          shift, fsize)[fslice].real

    return _centered(tmodel, freq.size)


def exp_model_log_likelihood(theta, data, inv_cov, freq, transfer):

    mdl = exp_model(theta, freq=freq, transfer=transfer)

    residual = data - mdl

    return -0.5 * np.matmul(residual.T, np.matmul(inv_cov, residual))


def exp_model_log_probability(theta, param_name, data, inv_cov, freq, transfer):

    lp = log_prior(theta, param_name=param_name)

    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + exp_model_log_likelihood(theta, data, inv_cov, freq, transfer)

