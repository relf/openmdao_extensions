"""
Driver for running model on design of experiments cases using OpenTURNS sampling methods
"""
import numpy as np

from openmdao.api import DOEDriver, OptionsDictionary
from openmdao.drivers.doe_generators import DOEGenerator

OPENTURNS_NOT_INSTALLED = False
try:
    import openturns as ot
except ImportError:
    OPENTURNS_NOT_INSTALLED = True

class OpenturnsDOEGenerator(DOEGenerator):
    def __init__(self):
        if OpenTURNS_NOT_INSTALLED:
            raise RuntimeError(
                "OpenTURNS library is not installed. cf. http://www.openturns.org/"
            )
        self.options.declare(
            "distribution",
            default=ot.Distribution,
            desc="Joint distribution of uncertain variables",
        )
        self.called = False

    def __call__(self, uncertain_vars, model=None):
        slef.dist = self.options["distribution"]
        if (len(uncertain_vars)) != (self.dist.getDimension()):
            raise RuntimeError(
                "Bad distribution dimension: should be equal to uncertain variables number {} "\
                    ", got {}".format(len(uncertain_vars), self.dist.getDimension())
            )

        self._compute_cases()
        self.called = True
        sample = []
        for i in range(self._cases.shape[0]):
            j = 0
            for name, meta in iteritems(design_vars):
                size = meta["size"]
                sample.append((name, self._cases[i, j : j + size]))
                j += size
            yield sample

    def _compute_cases(self):
        raise RuntimeError("Have to be implemented in subclass.")

    def get_cases(self):
        if not self.called:
            raise RuntimeError("Have to run the driver before getting cases")
        return self._cases

class OpenturnsLHSGenerator(DOEGenerator):
    def __init__(self, n_trajs=2, n_levels=4):
        super(OpenturnsLHSGenerator, self).__init__()
        # number of trajectories to apply morris method
        self.n_trajs = n_trajs
        # number of grid levels
        self.n_levels = n_levels

    def _compute_cases(self):
        self._cases = ms.sample(self._pb, self.n_trajs, self.n_levels)


class OpenturnsDOEDriver(DOEDriver):
    """
    Baseclass for OpenTURNS design-of-experiments Drivers
    """
    def __init__(self, **kwargs):
        super(OpenTurnsDOEDriver, self).__init__()
        self.options.declare(
            "doe_method_name",
            default="Morris",
            values=["Morris", "Sobol"],
            desc="either Morris or Sobol",
        )
        self.options.declare(
            "doe_options",
            types=dict,
            default={},
            desc="options for given OpenTURNS DOE method",
        )
        self.options.update(kwargs)
        self.doe_settings = OptionsDictionary()
        if self.options["sa_method_name"] == "Morris":
            self.sa_settings.declare(
                "n_trajs",
                types=int,
                default=2,
                desc="number of trajectories to apply morris method",
            )
            self.sa_settings.declare(
                "n_levels", types=int, default=4, desc="number of grid levels"
            )
            self.sa_settings.update(self.options["sa_doe_options"])
            n_trajs = self.sa_settings["n_trajs"]
            n_levels = self.sa_settings["n_levels"]
            self.options["generator"] = SalibMorrisDOEGenerator(n_trajs, n_levels)
        elif self.options["sa_method_name"] == "Sobol":
            self.sa_settings.declare(
                "n_samples",
                types=int,
                default=500,
                desc="number of samples to generate",
            )
            self.sa_settings.declare(
                "calc_second_order",
                types=bool,
                default=True,
                desc="calculate second-order sensitivities ",
            )
            self.sa_settings.update(self.options["sa_doe_options"])
            n_samples = self.sa_settings["n_samples"]
            calc_snd = self.sa_settings["calc_second_order"]
            self.options["generator"] = SalibSobolDOEGenerator(n_samples, calc_snd)
        else:
            raise RuntimeError(
                "Bad sensitivity analysis method name '{}'".format(
                    self.options["sa_method_name"]
                )
            )


    def _set_name(self):
        self._name = "OpenTURNS_DOE_" + self.options["doe_method_name"]

    def get_cases(self):
        return self.options["generator"].get_cases()
