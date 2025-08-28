# This contians some functions to work with MCMC chains of CHIME fitstack output

from typing import List, Dict, Tuple, Any, Union, Optional
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar
from getdist import MCSamples
from cora.signal import lssmodels
from cora.util import cosmology
from fitstack.containers import MCMCFit1D


def create_unified_config():
    """Create unified configuration dictionary.

    We are putting the redshift and redshift range for different tracer (cross-corr) and
    sub-bands (auto-ps) here as a dict. So that this can be loaded in later.
    If there is a new set of tracer for a new redshift range or a new subband for auto-ps
    analysis, just add that in the config accordingly.
    """
    cosmo = cosmology.Cosmology()

    config = {
        # Cross-correlation tracers and auto-corr bands
        "QSO": {
            "z_eff": 1.2034,
            "z_range": [1.0038, 1.365],
            "type": "cross",
            "bias_key": "eboss_qso",
        },
        "LRG": {
            "z_eff": 0.8372,
            "z_range": [0.8065, 0.8728],
            "type": "cross",
            "bias_key": "eboss_lrg",
        },
        "ELG": {
            "z_eff": 0.9576,
            "z_range": [0.8253, 1.0264],
            "type": "cross",
            "bias_key": "eboss_elg",
        },
        "QSOb0": {
            "z_eff": 0.9714,
            "z_range": [0.8501, 1.0073],
            "type": "cross",
            "bias_key": "eboss_qso",
        },
        "QSOb1": {
            "z_eff": 1.117,
            "z_range": [1.0651, 1.1631],
            "type": "cross",
            "bias_key": "eboss_qso",
        },
        "QSOb2": {
            "z_eff": 1.3028,
            "z_range": [1.2262, 1.3931],
            "type": "cross",
            "bias_key": "eboss_qso",
        },
        "QSOb00": {
            "z_eff": 0.8448,
            "z_range": [0.8179, 0.8666],
            "type": "cross",
            "bias_key": "eboss_qso",
        },
        "QSOb01": {
            "z_eff": 0.9857,
            "z_range": [0.9624, 1.0117],
            "type": "cross",
            "bias_key": "eboss_qso",
        },
        "QSObandb": {
            "z_eff": 1.159286,
            "z_range": [1.0006, 1.335],
            "type": "cross",
            "bias_key": "eboss_qso",
        },
        "bandb": {
            "z_eff": 1.159286,
            "z_range": [1.0006, 1.335],
            "type": "auto",
            "bias_key": "None",
        },
    }

    # Calculate properties for all entries
    for name, cfg in config.items():
        z = cfg["z_eff"]
        cfg.update(
            {
                "b_HI": 1 + lssmodels.bias["HI"](z),
                "omega_HI": lssmodels.omega_HI.evaluate(z),
                "f": cosmo.growth_rate(z),
            }
        )

        if cfg["type"] == "cross":
            cfg["b_g"] = 1 + lssmodels.bias[cfg["bias_key"]](z)

    return config


def scale_params(chain: MCSamples, identifier: str, config: Dict):
    """
    Scale the parameters actually fit to their values at the effective redshift.

    Parameters
    ----------
    chain : MCSamples
        MCMC samples chain to which derived parameters will be added.
    identifier : str
        Key used to extract properties from the configuration dictionary.
    config : dict
        Dictionary containing tracer or band properties.

    Returns
    -------
    None
        Modifies the input `chain` in place by adding scaled parameters.
    """
    props = config[identifier]

    chain.addDerived(
        chain["omega"] * props["omega_HI"] * 1e3,
        "omega_scaled",
        label=r"10^3 \times \Omega_\mathrm{HI}",
    )
    chain.addDerived(
        chain["b_HI"] * props["b_HI"], "b_HI_scaled", label=r"b_\mathrm{HI}"
    )

    if props["type"] == "cross":
        chain.addDerived(
            chain["b_g"] * props["b_g"], "b_g_scaled", label=r"b_\mathrm{g}"
        )


def _calc_omega_bHI_cov(
    c: MCSamples, config: Dict[str, Any], identifier: str, fscale: bool = False
):
    """
    Worker routine to calculate the omega_b_HI × omega covariance.

    Parameters
    ----------
    c : MCSamples
        MCMC samples chain containing derived parameters.
    config : dict
        Dictionary containing tracer or band properties.
    identifier : str
        Key to identify the tracer/band in config (e.g., 'QSO', 'bandb').
    fscale : bool, optional
        If True, scale by the growth rate `f`. Default is False.

    Returns
    -------
    ndarray
        2×2 covariance matrix of [omega_b_HI, omega].
    """
    omega = c["omega_scaled"]
    omega_b_HI = omega * c["b_HI_scaled"]

    if fscale:
        f = config[identifier]["f"]
        omega_b_HI /= f

    return np.cov([omega_b_HI, omega])


def calc_fmu2(
    c: MCSamples, config: Dict[str, Any], identifier: str, fscale: bool = False
) -> float:
    """
    Calculate the effective f μ² for the chain.

    Parameters
    ----------
    c : MCSamples
        MCMC samples chain containing derived parameters.
    config : dict
        Dictionary containing tracer or band properties.
    identifier : str
        Key to identify the tracer/band in config.
    fscale : bool, optional
        If True, scale by the growth rate `f`. Default is False.

    Returns
    -------
    float
        Effective fμ² value.
    """
    evals, evecs = la.eigh(_calc_omega_bHI_cov(c, config, identifier, fscale))
    return evecs[1, 0] / evecs[0, 0]


def calc_fmu2_single(
    chain_list: List[MCSamples],
    config: Dict[str, Any],
    identifier: List[str],
    fscale: bool = False,
) -> float:
    """
    Calculate a single best-fit f μ² value across multiple chains.

    Parameters
    ----------
    chain_list : list of MCSamples
        List of MCMC sample chains.
    config : dict
        Dictionary containing tracer or band properties.
    identifier : list of str
        List of identifiers corresponding to each chain.
    fscale : bool, optional
        If True, scale by the growth rate `f`. Default is False.

    Returns
    -------
    float
        Best-fit effective fμ² value.
    """
    iC = np.sum(
        [
            la.inv(_calc_omega_bHI_cov(c, config, identifier, fscale))
            for c in chain_list
        ],
        axis=0,
    )

    evals, evecs = la.eigh(iC)
    return evecs[1, 1] / evecs[0, 1]


def add_derived(
    c: MCSamples, fmu2: float, cross_corr: bool = True, sublabel: str = None
):
    """
    Add a set of useful derived parameters to the MCMC chain.

    Parameters
    ----------
    c : MCSamples
        MCMC samples chain where derived parameters are added.
    fmu2 : float
        Effective fμ² value to be included in derived calculations.
    cross_corr : bool
        If True derived parameters will be based on HI-galaxy cross-corr. Default is True.
    sublabel : str, optional
        Subscript label for derived parameter names. Default is None.

    Returns
    -------
    None
        Modifies the input `c` in place by adding derived parameters.
    """
    params = c.getParamNames().list()

    subscript = "" if sublabel is None else f"_{{{sublabel}}}"
    c.addDerived(
        c["omega_scaled"] * c["b_HI_scaled"],
        "omega_b_HI",
        label=r"10^3 \times \Omega_{\rm HI} b_{\rm HI}",
    )

    if cross_corr:
        if "FoGh" in params:

            c.addDerived(
                (c["FoGh"] * c["FoGg"]) ** 0.5, "FoG+", label=r"\alpha_\mathrm{FoG,+}"
            )
            c.addDerived(
                np.log(c["FoGh"] / c["FoGg"]), "FoG-", label=r"\alpha_\mathrm{FoG,-}"
            )

        # A_HI(stack) = (Omega_HI * b_HI + Omega_HI * <f mu^2>)
        c.addDerived(
            c["omega_scaled"] * c["b_HI_scaled"] + fmu2 * c["omega_scaled"],
            "omegabfm2",
            label=(r"\mathcal{A}_\mathrm{HI}" f"{subscript}"),
        )

    else:
        # A_HI(auto) = (Omega_HI * b_HI + Omega_HI * <f mu^2>)**2
        c.addDerived(
            (c["omega_scaled"] * c["b_HI_scaled"] + fmu2 * c["omega_scaled"]) ** 2,
            "omegabfm2",
            label=(r"\mathcal{A}_\mathrm{HI}" f"{subscript}"),
        )


def apply_limits(
    c: MCSamples,
    copy: bool = True,
    **kwargs,
) -> MCSamples:
    """
    Apply cutoff limits to a set of parameters in the MCMC chain.

    Parameters
    ----------
    c : MCSamples
        MCMC samples chain.
    copy : bool, optional
        If True, operate on a copy of the chain. Default is True.
    **kwargs : dict
        Parameter name → (lower, upper) cutoff values.

    Returns
    -------
    MCSamples
        Filtered MCMC chain with updated parameter ranges.
    """
    if copy:
        c = c.copy()

    params = c.getParamNames().list()

    for param in kwargs.keys():
        if param not in params:
            raise ValueError(
                f"Got keyword argument {param} that does not map to parameter."
            )

        lower, upper = kwargs[param]
        cond = (c[param] > lower) & (c[param] < upper)
        c.filter(cond)
        c.setRanges({param: (lower, upper)})

    return c


class MultiIndex(dict):
    """A dictionary indexed by multiple keys that can be queried and manipulated."""

    keynames: str
    _keyname_ind: Dict[str, int]

    def __init__(self, *keynames: str):
        self.keynames = keynames

        self._keyname_ind = {k: ii for ii, k in enumerate(keynames)}

    def __getitem__(self, key: Any):
        """Get the item indexed by the keys."""

        if not isinstance(key, tuple):
            key = (key,)

        return super().__getitem__(key)

    def __setitem__(self, key: Tuple, val: Any):

        # Validate key format
        if not isinstance(key, tuple) or len(key) != len(self.keynames):
            raise ValueError("Invalid key")

        super().__setitem__(key, val)

    def filter(self, **kwargs) -> "MultiIndex":
        """Filter values matching given values.

        Parameters
        ----------
        kwargs
            A series of keyname=keyvalue combinations for the filtering.

        Returns
        -------
        filtered
            A MultiIndex containing only the matching entries.
        """

        key_ind = []
        key_sel = []

        for keyname, sel in kwargs.items():
            if keyname not in self.keynames:
                raise KeyError("Invalid selection")

            key_ind.append(self._keyname_ind[keyname])
            key_sel.append(sel)

        new_keynames = tuple(kn for kn in self.keynames if kn not in kwargs.keys())

        new_md = self.__class__(*new_keynames)

        for k, v in self.items():

            for ind, val in zip(key_ind, key_sel):
                if k[ind] != val:
                    break
            else:
                newkeys = self._remove_key_ind(k, key_ind)
                new_md[newkeys] = v

        return new_md

    def groupkey(self, keyname: str) -> List[Tuple[Any, "MultiIndex"]]:
        """Return a list of pairs of a keyvalues, and all the matching items.

        Parameters
        ----------
        keyname
            The name of the key to group by.

        Returns
        -------
        results
            A list of pairs of each value within the keyname key, and a `MultiIndex`
            giving all the matching entries.
        """

        if keyname not in self.keynames:
            raise ValueError(f"Keyname={keyname} is not known.")

        ind = self._keyname_ind[keyname]

        # Extract all the possible values for keyname
        kvals = set(k[ind] for k in self.keys())

        return [(kval, self.filter(**{keyname: kval})) for kval in kvals]

    def fold(self, keyname: str) -> "MultiIndex":
        """Move keyname from the index into a list of pairs of keyvalues and items.

        Parameters
        ----------
        keyname
            The key name to move.

        Returns
        -------
        res
            A MultiIndex for all the remaining keynames. Each item is now a list of
            tuples of the keyvalues for the moved keyname, and the original items.
        """

        if keyname not in self.keynames:
            raise ValueError("Unknown keyname")

        key_ind = (self._keyname_ind[keyname],)

        new_md = MultiIndex(*self._remove_key_ind(self.keynames, key_ind))

        for k, v in self.items():
            k_val = k[key_ind[0]]
            new_k = self._remove_key_ind(k, key_ind)

            if new_k not in new_md:
                new_md[new_k] = []

            new_md[new_k].append((k_val, v))

        return new_md

    def _remove_key_ind(self, key: Tuple, ind: Tuple[int]):
        return tuple(kc for ii, kc in enumerate(key) if ii not in ind)


def hpd_chain(
    chain: Union[MCSamples, list[MCSamples]],
    param: str,
    lim: float = 0.68,
) -> Tuple[float, Tuple[float, float]]:
    """Get the highest posterior density estimate for param.

    Parameters
    ----------
    chain
        A single chain or a list of separate chains.
    param
        The parameter to estimate.
    lim
        The fraction within the credible interval.

    Returns
    -------
    mode
        Get the most likely value, i.e the distribution peak. This is an array if
        `chain` is a list.
    err
        A tuple of the negative and positive errors. This is a [2, nchain] array if
        `chain` is a list.
    """

    if isinstance(chain, list):
        modes, errs = zip(*[hpd_chain(c, param, lim) for c in chain])
        return np.array(modes), np.array(errs).T

    density = chain.get1DDensity(param)
    low, high, *_ = density.getLimits(lim)

    res = minimize_scalar(
        lambda x: -density.Prob(x), bounds=(low, high), method="bounded"
    )
    mode = res.x

    return mode, [mode - low, high - mode]


def cosmo_from_dict(cosmo: dict) -> cosmology.Cosmology:
    """Create a Cosmology instance from a dict of omega_m, omega_l, H0 parameters."""
    omega_m = cosmo.get("omega_m", 0.3)
    omega_l = cosmo.get("omega_l", 0.7)
    H0 = cosmo.get("H0", 70.0)
    return cosmology.Cosmology(omega_b=0, omega_c=omega_m, omega_l=omega_l, H0=H0)


def dXdz(z: float, cosmo: cosmology.Cosmology, delta_z: float = 1e-3) -> float:
    """Finite difference redshift derivative."""

    X1 = cosmo.comoving_distance(z + delta_z)
    X0 = cosmo.comoving_distance(z)

    return (X1 - X0) / delta_z


def cosmo_convert_HI(
    z: float, new_cosmo: cosmology.Cosmology, old_cosmo: cosmology.Cosmology
) -> float:
    """Construct the factor to convert an HI fractional density between cosmologies.

    The factor should multiply the old value.
    """

    # +2 powers of H0 from the critical density
    # -1 powers as the comoving distance is in Mpc/h And then a distance derivative
    # ratio (which turns out to be the same for both 21cm and DLAs)

    return (old_cosmo.H0 / new_cosmo.H0) * dXdz(z, old_cosmo) / dXdz(z, new_cosmo)


def combine_data(dset: dict, cosmo: cosmology.Cosmology = None) -> dict:
    """Extract and convert the Omega_HI data.

    Parameters
    ----------
    dset
        Omega_HI data formatted according to the schema.
    cosmo
        A common cosmology to convert the results to.

    Returns
    -------
    res
        A dictionary containing arrays of all the data in the dataset. Keys are:
        - `z`: the mean redshift
        - `zerr`: only present if a `z_range` was in the input. Shape [2, numz], giving
           the lower and upper error bars.
        - `omega_HI`: the actual data.
        - `omega_HI_err`: shape [2, numz] giving lower and upper error bars.
    """

    data_list = dset["data"]
    if not isinstance(data_list, list):
        data_list = [data_list]

    def parse_z(d):

        if "z" in d:
            z = d["z"]
        elif "z_range" in d:
            z = 0.5 * (d["z_range"][0] + d["z_range"][1])
        else:
            raise ValueError("Couldn't figure out a redshift")

        if "z_range" in d:
            zerr = (z - d["z_range"][0], d["z_range"][1] - z)
        else:
            zerr = None

        return z, zerr

    def parse_value(d):

        mul = 10 ** (-int(d["exp"]))

        value = d["value"][0] * mul

        # If there are multiple error terms combine them in quadrature, and turn into a
        # two sided error spec
        if len(d["value"]) > 2:
            err = np.sum(np.array(d["value"][1:]) ** 2) ** 0.5 * mul
            err = (err, err)

        # Deal with two sided errors (note that the errors in the files are reverse
        # compared to matplotlib)
        elif isinstance(d["value"][1], list):
            err = (mul * d["value"][1][1], mul * d["value"][1][0])

        else:
            err = (mul * d["value"][1], mul * d["value"][1])

        return value, err

    z, zerr = zip(*[parse_z(d) for d in data_list])

    z = np.array(z)

    if zerr[0] is None:
        zerr = None
    else:
        zerr = np.array(zerr).T

    if cosmo is not None:
        old_cosmo = cosmo_from_dict(dset["cosmo"])
        conversion_factor = cosmo_convert_HI(z, cosmo, old_cosmo)
    else:
        conversion_factor = 1.0

    if "DLA" in dset["type"]:
        conversion_factor /= dset.get("mu", 1.0)

    omega_HI, omega_HI_err = zip(*[parse_value(d) for d in data_list])
    omega_HI = np.array(omega_HI) * conversion_factor
    omega_HI_err = np.array(omega_HI_err).T * conversion_factor

    return dict(z=z, zerr=zerr, omega_HI=omega_HI, omega_HI_err=omega_HI_err)


def extract_priors(chain: MCMCFit1D) -> Dict[str, Tuple[float, float]]:
    """
    Extract uniform priors from a chain file.

    Parameters
    ----------
    chain : MCMCFit1D
        MCMC chain object containing configuration and parameter information.

    Returns
    -------
    dict of str → (float, float)
        Dictionary mapping parameter names to `(low, high)` prior ranges.
        Returns an empty dictionary if no configuration is found.
    """
    if "config" not in chain.history:
        return {}

    task_list = chain.history["config"]["pipeline"]["tasks"]

    mcmc_task = [t for t in task_list if "RunMCMC" in t["type"]]

    if len(mcmc_task) != 1:
        raise ValueError(
            f"Config must have only one RunMCMC task. Found {len(mcmc_task)}."
        )

    params = mcmc_task[0].get("params", {}).get("param_spec", {})

    priors = {}
    for pname, pspec in params.items():

        if pspec["prior"] != "Uniform" or pspec.get("fixed", False):
            continue

        low = pspec["kwargs"]["low"]
        high = pspec["kwargs"]["high"]

        priors[pname] = (low, high)

    return priors


def gdchain(
    chain: MCMCFit1D,
    latex_names: Optional[Union[list, dict]] = None,
    thin: Optional[int] = None,
    **kwargs,
) -> MCSamples:
    """
    Convert a CHIME MCMCFit object into a `getdist.MCSamples` object.

    Parameters
    ----------
    chain : MCMCFit1D
        MCMC chain object containing datasets and parameter mappings.
    latex_names : list or dict, optional
        Either:
          - A list/tuple of LaTeX-style labels for parameters, matching the number of parameters.
          - A dictionary mapping parameter names to LaTeX labels.
          - If None, parameter names are used as-is.
    thin : int, optional
        Thinning factor to reduce the number of samples. Default is None (no thinning).
    **kwargs : dict
        Additional keyword arguments passed to `MCSamples`.

    Returns
    -------
    MCSamples
        A `getdist.MCSamples` object containing the chain samples and parameter labels.
    """
    samples = list(chain.datasets["chain"][:].transpose(1, 0, 2))
    loglikes = list(chain.datasets["chisq"][:].T / 2)
    params = chain.index_map["param"][:]

    if latex_names is not None:

        if isinstance(latex_names, (tuple, list)):
            if len(latex_names) != len(params):
                raise ValueError(
                    "Latex names must be a dict, or a list with the same length as the"
                    "number of parameters."
                )
            kwargs["labels"] = latex_names

        if isinstance(latex_names, dict):
            kwargs["labels"] = [latex_names.get(param, param) for param in params]

    samples = MCSamples(samples=samples, names=params, loglikes=loglikes, **kwargs)

    if thin:
        samples.thin(thin)

    return samples
