"""Routines for chi^2 tests of stacking measurements.

Used for CHIME-eBOSS stacking analysis in arXiv:2202.01242.
"""

import logging
import inspect

import numpy as np
import scipy.optimize

from caput import config, pipeline

from draco.core import task

from . import containers
from . import utils
from . import models
from . import priors

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)]
    )


SIMULATION_MODELS = [
    c.__name__
    for c in [models.SimulationTemplate]
    + list(_all_subclasses(models.SimulationTemplate))
]

BOUNDED_MINIMIZATION = ["L-BFGS-B", "Nelder-Mead", "Powell", "TNC"]


def get_param0(fit, param_to_fit):

    chisq = fit["chisq"][:]
    chain = fit["chain"][:]
    param = list(fit.index_map["param"][:])

    imin = np.argmin(np.abs(chisq.flatten()))
    theta_min = chain.reshape(-1, chain.shape[-1])[imin]

    ifit = np.array([param.index(pfit) for pfit in param_to_fit])

    return theta_min[ifit]


def get_bounds(mdl, scale_bound=0.0):

    param = mdl.param_name_fit
    nparam = len(param)

    lb = np.zeros(nparam, dtype=np.float64)
    ub = np.zeros(nparam, dtype=np.float64)

    for nn, name in enumerate(param):

        prior = mdl.priors[name]

        if isinstance(prior, priors.Uniform):
            lb[nn] = prior.low
            ub[nn] = prior.high
        else:
            lb[nn] = prior.loc - 5.0 * prior.scale
            ub[nn] = prior.loc + 5.0 * prior.scale

        if scale_bound > 0.0:
            db = ub[nn] - lb[nn]
            lb[nn], ub[nn] = lb[nn] - scale_bound * db, ub[nn] + scale_bound * db

    return scipy.optimize.Bounds(lb, ub)


def chisq_test(
    restricted,
    unrestricted,
    model_kwargs=None,
    param_spec=None,
    method="L-BFGS-B",
    options=None,
    scale_bound=0.0,
    required_pol=None,
):

    if required_pol is None:
        required_pol = ["XX", "YY"]

    if model_kwargs is None:
        model_kwargs = {"restricted": {}, "unrestricted": {}}

    if param_spec is None:
        param_spec = {"restricted": {}, "unrestricted": {}}

    if options is None:
        options = {}

    fr = containers.MCMCFit1D.from_file(utils.find_file(restricted))
    fu = containers.MCMCFit1D.from_file(utils.find_file(unrestricted))

    pol_sel = np.array([list(fr.pol).index(pstr) for pstr in required_pol])

    for key in ["restricted", "unrestricted"]:
        model_kwargs[key]["pol"] = ["XX", "YY"]
        model_kwargs[key]["combine"] = False
        model_kwargs[key]["sort"] = True

    fit_kwargs = {}
    fit_kwargs["freq"] = fr.freq
    fit_kwargs["data"] = fr.stack[pol_sel]
    fit_kwargs["inv_cov"] = fr["precision"][:]
    fit_kwargs["transfer"] = None

    eval_kwargs_r = {"freq": fr.freq, "transfer": None}

    # Prepare the model
    namer = fr.attrs["model"]
    Modelr = getattr(models, namer)
    modelr = Modelr(**{**model_kwargs["restricted"], **param_spec["restricted"]})

    nameu = fu.attrs["model"]
    Modelu = getattr(models, nameu)
    modelu = Modelu(**{**model_kwargs["unrestricted"], **param_spec["unrestricted"]})

    # Specialized fit keywords
    fit_kwargs_r = {key: val for key, val in fit_kwargs.items()}
    if (
        (namer in SIMULATION_MODELS)
        and ("DualPol" not in namer)
        and ("Split" not in namer)
    ):
        fit_kwargs_r["pol_sel"] = pol_sel
        eval_kwargs_r["pol_sel"] = pol_sel

    fit_kwargs_u = {key: val for key, val in fit_kwargs.items()}
    if (
        (nameu in SIMULATION_MODELS)
        and ("DualPol" not in nameu)
        and ("Split" not in nameu)
    ):
        fit_kwargs_u["pol_sel"] = pol_sel

    # Construct data realizations
    mr = fr["mock"][:, pol_sel]

    nmock, npol, nfreq = mr.shape

    # Create output container
    out = containers.ChisqTest(
        mock=np.arange(nmock, dtype=np.int),
        restricted_param=np.array(modelr.param_name_fit),
        unrestricted_param=np.array(modelu.param_name_fit),
    )

    # Fit the restricted model
    # --------------------------
    modelr.set_data(**fit_kwargs_r)

    if modelr.nfit > 0:

        # Determine the starting point and parameter bounds
        param0r = get_param0(fr, modelr.param_name_fit)
        nparamr = param0r.size
        if method in BOUNDED_MINIMIZATION:
            boundr = get_bounds(modelr, scale_bound=scale_bound)
        else:
            boundr = None

        resdr = scipy.optimize.minimize(
            modelr.negative_log_likelihood,
            param0r,
            method=method,
            bounds=boundr,
            options=options,
        )

        out.attrs["restricted_success"] = resdr.success
        out.attrs["restricted_chisq"] = 2.0 * modelr.negative_log_likelihood(resdr.x)
        out.attrs["restricted_param"] = resdr.x

        thetar = modelr.get_all_params(resdr.x)

        out.add_dataset("restricted_param")
        paramr = out["restricted_param"][:].view(np.ndarray)

    else:

        out.attrs["restricted_success"] = True
        out.attrs["restricted_chisq"] = 2.0 * modelr.negative_log_likelihood([])

        thetar = modelr.get_all_params([])

    yr = modelr.model(thetar, **eval_kwargs_r)[np.newaxis, ...] + mr

    # Fit the unrestricted model
    # --------------------------
    modelu.set_data(**fit_kwargs_u)

    # Determine the starting point and parameter bounds
    param0u = get_param0(fu, modelu.param_name_fit)
    nparamu = param0u.size
    if method in BOUNDED_MINIMIZATION:
        boundu = get_bounds(modelu, scale_bound=scale_bound)
    else:
        boundu = None

    resdu = scipy.optimize.minimize(
        modelu.negative_log_likelihood,
        param0u,
        method=method,
        bounds=boundu,
        options=options,
    )

    out.attrs["unrestricted_success"] = resdu.success
    out.attrs["unrestricted_chisq"] = 2.0 * modelu.negative_log_likelihood(resdu.x)
    out.attrs["unrestricted_param"] = resdu.x

    # Dereference datasets
    successr = out["restricted_success"][:].view(np.ndarray)
    chisqr = out["restricted_chisq"][:].view(np.ndarray)

    successu = out["unrestricted_success"][:].view(np.ndarray)
    paramu = out["unrestricted_param"][:].view(np.ndarray)
    chisqu = out["unrestricted_chisq"][:].view(np.ndarray)

    # Loop over mocks
    for mm in range(nmock):

        # Print progress update to log
        if (mm % 100) == 0:
            logger.info(f"Fitting data realization {mm} of {nmock}.")

        # Fit restricted model
        fit_kwargs_r["data"] = yr[mm]
        modelr.set_data(**fit_kwargs_r)

        if modelr.nfit > 0:
            resr = scipy.optimize.minimize(
                modelr.negative_log_likelihood,
                param0r,
                method=method,
                bounds=boundr,
                options=options,
            )

            successr[mm] = resr.success
            paramr[mm] = resr.x
            chisqr[mm] = 2.0 * modelr.negative_log_likelihood(resr.x)

        else:
            successr[mm] = True
            chisqr[mm] = 2.0 * modelr.negative_log_likelihood(modelr.default_values)

        # Fit unrestricted model
        fit_kwargs_u["data"] = yr[mm]
        modelu.set_data(**fit_kwargs_u)

        resu = scipy.optimize.minimize(
            modelu.negative_log_likelihood,
            param0u,
            method=method,
            bounds=boundu,
            options=options,
        )

        successu[mm] = resu.success
        paramu[mm] = resu.x
        chisqu[mm] = 2.0 * modelu.negative_log_likelihood(resu.x)

    # Return the output container
    return out


class ChisqTest(task.SingleTask):
    """Pipeline task that calls the chisq_test function.

    Enables the user to call the chisq_test method with caput-pipeline,
    which provides many useful features including profiling, job script
    generation, job templating, and saving the results to disk.

    Attributes
    ----------
    max_iter : int
        Number of times to call the chisq_test method.
        Defaults to 1.

    See the arguments of the chisq_test method for a list of
    additional attributes and their default values.
    """

    max_iter = config.Property(proptype=int, default=1)

    restricted = config.Property(proptype=str)
    unrestricted = config.Property(proptype=str)
    param_spec = config.Property(proptype=dict)
    model_kwargs = config.Property(proptype=dict)
    options = config.Property(proptype=dict)
    method = config.Property(proptype=str)
    scale_bound = config.Property(proptype=float)
    required_pol = config.Property(proptype=list)

    def setup(self):
        """Prepare all arguments to the chisq_test function."""

        # Use the default values from the chisq_test method,
        # so we do not have to repeat them in two places.
        signature = inspect.signature(chisq_test)
        defaults = {
            k: v.default if v.default is not inspect.Parameter.empty else None
            for k, v in signature.parameters.items()
        }

        self.kwargs = {}
        for key, default_val in defaults.items():
            if hasattr(self, key):
                prop_val = getattr(self, key)
                self.kwargs[key] = prop_val if prop_val is not None else default_val
            else:
                self.log.warning(
                    "ChisqTest does not have a property corresponding "
                    f"to the {key} keyword argument to chisq_test."
                )

    def process(self):
        """Fit a model to the source stack using an MCMC."""

        if self._count == self.max_iter:
            raise pipeline.PipelineStopIteration

        result = chisq_test(**self.kwargs)

        return result
