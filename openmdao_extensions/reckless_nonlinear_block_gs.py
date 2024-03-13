"""Define the NonlinearBlockGS class.

This class was derived from OpenMDAO 3.30 NonlinearBlockGS code 
"""

import os
import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.api import NonlinearBlockGS


class RecklessNonlinearBlockGS(NonlinearBlockGS):
    """
    Extends Nonlinear block Gauss-Seidel solver with convergence variables options.
    Those options allows to focus on a subset of variables to drive the convergence.
    It allows to get quickest convergence by ignoring 'noise' coming highly non linear
    variables. Obviously the user has to know what he/she is doing because
    in that case some of the variables may not be converged properly
    (hence the 'reckless' prefix in the name).

    Attributes
    ----------
    _convrg_vars: list of string
        List of absolute variable names used to compute relative error and control
        solver convergence.
    _convrg_rtols: list of float
        List of relative error tolerance values for each variables of _convrg_vars. If not set, rtol
        value is used for all specified variables. Only used if _convrg_vars is set.
    """

    SOLVER = "NL: RNLBGS"

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super().__init__(**kwargs)

        self._convrg_vars = None
        self._convrg_rtols = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()
        self.options.declare(
            "convrg_vars",
            types=list,
            default=[],
            desc="list of variables (names) used by relative error criterium.",
        )
        self.options.declare(
            "convrg_rtols",
            types=list,
            default=[],
            desc="list of relative error tolerances corresponding to each"
            " variable specified in convrg_vars option (rtol is used otherwise)",
        )

    def _solve(self):
        """
        Run the iterative solver.

        Overrides opendmao/solvers/solver.py to implement _is_rtol_converged
        """
        maxiter = self.options["maxiter"]
        atol = self.options["atol"]
        iprint = self.options["iprint"]

        self._mpi_print_header()

        self._iter_count = 0
        self._iter_initialize()

        if self._convrg_vars:
            # get initial value of convrg_vars
            norm, val_convrg_vars_minus1 = self._iter_get_norm()
            ratio = 1
        else:
            norm = self._iter_get_norm()
            norm0 = norm if norm != 0.0 else 1.0
            ratio = norm / norm0

        self._mpi_print(self._iter_count, norm, ratio)
        is_rtol_converged = self._is_rtol_converged(ratio)

        while self._iter_count < maxiter and norm > atol and not is_rtol_converged:
            with Recording(type(self).__name__, self._iter_count, self) as rec:
                self._single_iteration()
                self._iter_count += 1
                self._run_apply()

                if self._convrg_vars:
                    norm, val_convrg_vars = self._iter_get_norm()
                    rec.abs = norm
                    rec.rel = (
                        np.abs(val_convrg_vars - val_convrg_vars_minus1)
                        / val_convrg_vars_minus1
                    )
                    ratio = (
                        np.abs(val_convrg_vars - val_convrg_vars_minus1)
                        / val_convrg_vars_minus1
                    )
                    ratio = np.max(ratio)

                    val_convrg_vars_minus1 = val_convrg_vars
                else:
                    norm = self._iter_get_norm()
                    # With solvers, we want to record the norm AFTER the call, but the call needs to
                    # be wrapped in the with for stack purposes, so we locally assign  norm & norm0
                    # into the class.
                    rec.abs = norm
                    rec.rel = norm / norm0
                    ratio = norm / norm0
                    if norm0 == 0:
                        norm0 = 1

            self._mpi_print(self._iter_count, norm, ratio)
            is_rtol_converged = self._is_rtol_converged(ratio)

        system = self._system()
        if system.comm.rank == 0 or os.environ.get("USE_PROC_FILES"):
            prefix = self._solver_info.prefix + self.SOLVER
            is_rtol_converged = self._is_rtol_converged(ratio)
            # Solver terminated early because a Nan in the norm doesn't satisfy the while-loop
            # conditionals.
            if np.isinf(norm) or np.isnan(norm):
                msg = (
                    "Solver '{}' on system '{}': residuals contain 'inf' or 'NaN' after {} "
                    + "iterations."
                )
                if iprint > -1:
                    print(
                        prefix
                        + msg.format(self.SOLVER, system.pathname, self._iter_count)
                    )

                # Raise AnalysisError if requested.
                if self.options["err_on_non_converge"]:
                    raise AnalysisError(
                        msg.format(self.SOLVER, system.pathname, self._iter_count)
                    )

            # Solver hit maxiter without meeting desired tolerances.
            elif norm > atol and not is_rtol_converged:
                msg = "Solver '{}' on system '{}' failed to converge in {} iterations."

                if iprint > -1:
                    print(
                        prefix
                        + msg.format(self.SOLVER, system.pathname, self._iter_count)
                    )

                # Raise AnalysisError if requested.
                if self.options["err_on_non_converge"]:
                    raise AnalysisError(
                        msg.format(self.SOLVER, system.pathname, self._iter_count)
                    )

            # Solver converged
            elif iprint == 1:
                print(prefix + " Converged in {} iterations".format(self._iter_count))
            elif iprint == 2:
                print(prefix + " Converged")

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        self._convrg_vars = self.options["convrg_vars"]
        if self._convrg_vars and not self.options["convrg_rtols"]:
            rtol = self.options["rtol"]
            self._convrg_rtols = rtol * np.ones(len(self._convrg_vars))
        else:
            self._convrg_rtols = self.options["convrg_rtols"]
            if len(self._convrg_rtols) != len(self._convrg_vars):
                raise RuntimeError(
                    "Convergence rtols bad size : should be {}, " "found {}.".format(
                        len(self._convrg_vars), len(self._convrg_rtols)
                    )
                )

        return super()._iter_initialize()

    def _is_rtol_converged(self, ratio):
        """
        Check convergence regarding relative error tolerance.

        Parameters
        ----------
        norm : float
            error (residuals norm)
        norm0 : float
            initial error

        Returns
        -------
        bool
            whether convergence is reached regarding relative error tolerance
        """
        system = self._system()
        if self._convrg_vars:
            nbvars = len(self._convrg_vars)
            rerrs = np.ones(nbvars)
            outputs = np.ones(nbvars)
            for i, name in enumerate(self._convrg_vars):
                outputs = system._outputs._views[name]
                residual = system._residuals._views[name]
                rerrs[i] = np.linalg.norm(residual) / np.linalg.norm(outputs)
            is_rtol_converged = (rerrs < self._convrg_rtols).all()
            is_rtol_converged = ratio < self.options["rtol"]
        else:
            is_rtol_converged = ratio < self.options["rtol"]
        return is_rtol_converged

    def _iter_get_norm(self):
        """
        Return the norm of the residual regarding convergence variable settings.

        Returns
        -------
        float
            norm.
        """
        system = self._system()
        if self._convrg_vars:
            total = []
            val_convrg_vars = np.zeros(len(self._convrg_vars))
            for i, name in enumerate(self._convrg_vars):
                total.append(system._residuals._views_flat[name])
                val_convrg_vars[i] = np.linalg.norm(system._outputs._views[name])
            norm = np.linalg.norm(np.concatenate(total))
        else:
            norm = super(RecklessNonlinearBlockGS, self)._iter_get_norm()

        if self._convrg_vars:
            return norm, val_convrg_vars
        else:
            return norm
