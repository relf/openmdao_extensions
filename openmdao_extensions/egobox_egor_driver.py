import numpy as np
import traceback

from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.core.analysis_error import AnalysisError

EGOBOX_NOT_INSTALLED = False
try:
    import egobox as egx
    from egobox import Egor, GpConfig, TregoConfig, QEiConfig
except ImportError:
    EGOBOX_NOT_INSTALLED = True


def to_list(lst, size):
    if not (isinstance(lst, np.ndarray) or isinstance(lst, list)):
        return [lst] * size
    diff_len = len(lst) - size
    if diff_len > 0:
        return lst[0:size]
    elif diff_len < 0:
        return [lst[0]] * size
    else:
        return lst


class EgoboxEgorDriver(Driver):
    """OpenMDAO driver for egobox optimizer"""

    def __init__(self, **kwargs):
        """Initialize the driver with the given options."""
        super(EgoboxEgorDriver, self).__init__(**kwargs)

        if EGOBOX_NOT_INSTALLED:
            raise RuntimeError("egobox library is not installed.")

        # What we support
        self.supports["optimization"] = True
        self.supports["inequality_constraints"] = True
        self.supports["linear_constraints"] = True
        self.supports["integer_design_vars"] = True

        # What we don't support
        self.supports["equality_constraints"] = False
        self.supports["two_sided_constraints"] = False
        self.supports["multiple_objectives"] = False
        self.supports["active_set"] = False
        self.supports["simultaneous_derivatives"] = False
        self.supports["total_jac_sparsity"] = False
        self.supports["gradients"] = False
        self.supports._read_only = True

        self.opt_settings = {}

    def _declare_options(self):
        self.options.declare(
            "optimizer",
            default="EGOR",
            values=["EGOR"],
            desc="Name of optimizer to use",
        )

    def _setup_driver(self, problem):
        super(EgoboxEgorDriver, self)._setup_driver(problem)

        self.comm = None

    def run(self):
        model = self._problem().model

        self.iter_count = 0
        self.name = f"egobox_optimizer_{self.options['optimizer'].lower()}"

        # Initial Run
        with RecordingDebugging(self.name, self.iter_count, self) as rec:
            # Initial Run
            model.run_solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        # Format design variables to suit segomoe implementation
        self.xspecs = self._initialize_vars(model)

        # Format constraints to suit segomoe implementation
        self.n_cstr = self._initialize_cons()

        # Format option dictionary to suit Egor implementation
        cstr_tol = [1e-4] * self.n_cstr if self.n_cstr else []
        optim_settings = {
            "cstr_tol": cstr_tol,
        }
        n_iter = self.opt_settings["maxiter"]

        # Filter out options requiring an object and ignore run options
        optim_settings.update(
            {
                k: v
                for k, v in self.opt_settings.items()
                if k
                not in [
                    "gp_config",
                    "trego",
                    "qei_config",
                    "maxiter",
                    "run_info",
                    "fcstrs",
                ]
            }
        )

        # Manage gp_config special case: conf object GpConfig
        gp_config_args = self.opt_settings.get("gp_config", {})

        # Manage trego_config special case: conf object TregoConfig
        trego_config_args = self.opt_settings.get("trego", {})

        # Manage qei_config special case: conf object QeiConfig
        qei_config_args = self.opt_settings.get("qei_config", {})

        dim = 0
        for name, meta in self._designvars.items():
            dim += meta["size"]
        if dim > 10:
            self.optim_settings["kpls_dim"] = 3

        # Instanciate a SEGO optimizer
        egor = Egor(
            xspecs=self.xspecs,
            gp_config=GpConfig(**gp_config_args),
            qei_config=QEiConfig(**qei_config_args),
            trego=TregoConfig(**trego_config_args),
            n_cstr=self.n_cstr,
            **optim_settings,
        )

        # Run the optim
        res = egor.minimize(self._objfunc, max_iters=n_iter)

        # Set optimal parameters
        i = 0
        for name, meta in self._designvars.items():
            size = meta["size"]
            self.set_design_var(name, res.x_opt[i : i + size])
            i += size

        with RecordingDebugging(self.name, self.iter_count, self) as rec:
            model.run_solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0
        self.iter_count += 1

        return True

    def _initialize_vars(self, model):
        dvs_int = {}
        for name, meta in self._designvars.items():
            infos = model.get_io_metadata(includes=name)
            for absname in infos:
                if name == infos[absname]["prom_name"] and (
                    infos[absname]["tags"] & {"wop:int"}
                ):
                    dvs_int[name] = egx.XType.INT

        variables = []
        desvars = self._designvars
        for name, meta in desvars.items():
            vartype = dvs_int.get(name, egx.XType.FLOAT)
            size = meta["size"]
            meta_low = meta["lower"]
            meta_high = meta["upper"]
            if size > 1:
                for j in range(size):
                    if isinstance(meta_low, np.ndarray):
                        p_low = meta_low[j]
                    else:
                        p_low = meta_low

                    if isinstance(meta_high, np.ndarray):
                        p_high = meta_high[j]
                    else:
                        p_high = meta_high

                    variables += [egx.XSpec(vartype, [p_low, p_high])]
            else:
                variables += [egx.XSpec(vartype, [meta_low, meta_high])]
        return variables

    def _initialize_cons(self, eq_tol=None, ieq_tol=None):
        """Format OpenMDAO constraints to suit EGOR implementation

        Parameters
        ----------
        eq_tol: dict
            Dictionary to define specific tolerance for eq constraints
            {'[groupName]': [tol]} Default tol = 1e-5
        """
        con_meta = self._cons

        self.ieq_cons = {
            name: con for name, con in con_meta.items() if not con["equals"]
        }

        # Inequality constraints
        n_cstr = 0
        for name in self.ieq_cons.keys():
            meta = con_meta[name]
            size = meta["size"]
            # Bounds - double sided is supported
            lower = to_list(meta["lower"], size)
            upper = to_list(meta["upper"], size)
            for k in range(size):
                if (lower[k] is None or lower[k] < -1e29) and upper[k] == 0.0:
                    n_cstr += 1
                else:
                    raise ValueError(
                        f"Constraint {lower[k]} < g(x) < {upper[k]} not handled by Egor driver"
                    )
        return n_cstr

    def _objfunc(self, points):
        """
        Function that evaluates and returns the objective function and the
        constraints. This function is called by SEGOMOE

        Parameters
        ----------
        point : numpy.ndarray
            point to evaluate

        Returns
        -------
        func_dict : dict
            Dictionary of all functional variables evaluated at design point.
        fail : int
            0 for successful function evaluation
            1 for unsuccessful function evaluation
        """
        res = np.zeros((points.shape[0], 1 + self.n_cstr))
        model = self._problem().model

        for k, point in enumerate(points):
            try:
                # Pass in new parameters
                i = 0

                for name, meta in self._designvars.items():
                    size = meta["size"]
                    self.set_design_var(name, point[i : i + size])
                    i += size

                # Execute the model
                with RecordingDebugging(
                    self.options["optimizer"], self.iter_count, self
                ) as _:
                    self.iter_count += 1
                    try:
                        model.run_solve_nonlinear()

                    # Let the optimizer try to handle the error
                    except AnalysisError:
                        model._clear_iprint()

                # Get the objective function evaluation - single obj support
                for obj in self.get_objective_values().values():
                    res[k, 0] = obj.item()

                # Get the constraint evaluations
                j = 1
                for con_res in self.get_constraint_values().values():
                    # Make sure con_res is array_like
                    con_res = to_list(con_res, 1)
                    # Perform mapping
                    for i, _ in enumerate(con_res):
                        res[k, j + i] = con_res[i]
                    j += 1

            except Exception as msg:
                tb = traceback.format_exc()
                print("Exception: %s" % str(msg))
                print(70 * "=", tb, 70 * "=")

        return res
