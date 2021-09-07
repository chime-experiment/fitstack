"""Define statistical distributions that can be used to place priors on parameters."""

import numpy as np


class Prior(object):
    """Base class representing a prior distribution."""

    def __init__(self, rng=None, seed=None, **kwargs):
        """Initialize the random number generator.

        Parameters
        ----------
        rng: np.random.Generator
            Random number generator.
        seed : int
            Seed to use for random number generation.  Only used if rng
            was not provided.  If the seed is not provided, then a random
            seed will be taken from system entropy.
        """

        if rng is None:

            if seed is None:
                seed = np.random.SeedSequence().entropy

            self.rng = np.random.Generator(np.random.SFC64(seed))

        else:
            self.rng = rng


class Uniform(Prior):
    """Uniform prior distribution."""

    def __init__(self, low=0.0, high=1.0, **kwargs):
        """Set the lower and upper boundaries of the distribution.

        Parameters
        ----------
        low : float
            The lower boundary.
        high : float
            The upper boundary.
        """

        super().__init__(**kwargs)

        self.low = low
        self.high = high
        self.norm = 1.0 / (high - low)

    def evaluate(self, theta):
        """Evaluate the probability of observing this value of the parameter.

        Parameters
        ----------
        theta : float
            The parameter value.

        Returns
        -------
        prob : float
            The probability of observing the input parameter value.
        """

        return self.norm * ((theta >= self.low) and (theta <= self.high))

    def draw_random(self):
        """Draw a random value of the parameter from the prior distribution.

        Returns
        -------
        theta : float
            Random value of the parameter.
        """

        return self.rng.uniform(low=self.low, high=self.high)


class Gaussian(Prior):
    """Gaussian prior distribution."""

    def __init__(self, loc=0.0, scale=1.0, **kwargs):
        """Set the location and scale of the distribution.

        Parameters
        ----------
        loc : float
            The mean.
        scale : float
            The standard deviation.
        """

        super().__init__(**kwargs)

        self.loc = loc
        self.scale = scale
        self.norm = 1.0 / np.sqrt(2.0 * np.pi * self.scale ** 2)

    def evaluate(self, theta):
        """Evaluate the probability of observing this value of the parameter.

        Parameters
        ----------
        theta : float
            The parameter value.

        Returns
        -------
        prob : float
            The probability of observing the input parameter value.
        """

        return self.norm * np.exp(-((theta - loc) ** 2) / (2.0 * self.scale ** 2))

    def draw_random(self):
        """Draw a random value of the parameter from the prior distribution.

        Returns
        -------
        theta : float
            Random value of the parameter.
        """

        return self.rng.normal(loc=self.loc, scale=self.scale)
