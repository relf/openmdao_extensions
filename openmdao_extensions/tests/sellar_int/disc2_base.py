"""
  disc2_base.py generated by WhatsOpt 1.25.1
"""
# DO NOT EDIT unless you know what you are doing
# whatsopt_url: http://localhost:3000
# analysis_id: 4


import numpy as np
from numpy import nan, inf
import openmdao.api as om


class Disc2Base(om.ExplicitComponent):
    """An OpenMDAO base component to encapsulate Disc2 discipline.
    This class defines inputs and outputs of the discipline and declare partials.
    """

    def setup(self):
        self.add_input("y1", val=1.0, desc="")
        self.add_input("z", val=[5, 2], desc="")

        self.add_output("y2", val=1.0, desc="")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")