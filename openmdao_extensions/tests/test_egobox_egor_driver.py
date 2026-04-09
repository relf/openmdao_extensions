import os
import unittest
import openmdao.api as om
from openmdao.test_suite.components.sellar_feature import SellarMDA
from openmdao_extensions.egobox_egor_driver import EgoboxEgorDriver
from openmdao_extensions.egobox_egor_driver import EGOBOX_NOT_INSTALLED

from openmdao_extensions.tests.functions_test import BraninMDA, AckleyMDA


class TestEgor(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @unittest.skipIf(EGOBOX_NOT_INSTALLED, "egobox is not installed")
    def test_sellar(self):
        self.pb = pb = om.Problem(SellarMDA())
        pb.model.add_design_var("x", lower=0, upper=10)
        pb.model.add_design_var("z", lower=0, upper=10)
        pb.model.add_objective("obj")
        pb.model.add_constraint("con1", upper=0)
        pb.model.add_constraint("con2", upper=0)
        pb.driver = EgoboxEgorDriver(optimizer="EGOR")
        pb.driver.opt_settings["maxiter"] = 10
        pb.driver.opt_settings["verbose"] = 2
        pb.driver.opt_settings["seed"] = 42
        pb.driver.opt_settings["timeout"] = 0.200
        pb.driver.opt_settings["outdir"] = "./test_sellar_out"
        pb.setup()
        self.case_recorder_filename = "{}/test_egobox_driver_sellar.sqlite".format(
            pb.get_outputs_dir()
        )
        recorder = om.SqliteRecorder(self.case_recorder_filename)
        pb.model.add_recorder(recorder)
        self.pb.run_driver()
        self.assertTrue(os.path.exists(self.case_recorder_filename))
        reader = om.CaseReader(self.case_recorder_filename)
        for case_id in reader.list_cases():
            case = reader.get_case(case_id)
            print(case.outputs["obj"])

    @unittest.skipIf(EGOBOX_NOT_INSTALLED, "egobox is not installed")
    def test_sellar_int(self):
        self.pb = pb = om.Problem(SellarMDA())
        pb.model.add_design_var("x", lower=0, upper=10)
        pb.model.add_design_var("z", lower=0, upper=10)
        pb.model.add_objective("obj")
        pb.model.add_constraint("con1", upper=0)
        pb.model.add_constraint("con2", upper=0)
        pb.driver = EgoboxEgorDriver(optimizer="EGOR")
        pb.driver.opt_settings["maxiter"] = 10

        pb.setup()
        self.case_recorder_filename = "{}/test_egobox_driver_sellar_int.sqlite".format(
            pb.get_outputs_dir()
        )
        recorder = om.SqliteRecorder(self.case_recorder_filename)
        pb.model.add_recorder(recorder)
        self.pb.run_driver()
        self.assertTrue(os.path.exists(self.case_recorder_filename))
        reader = om.CaseReader(self.case_recorder_filename)
        for case_id in reader.list_cases():
            case = reader.get_case(case_id)
            print(f"obj = {case.outputs['obj']}")

    @unittest.skipIf(EGOBOX_NOT_INSTALLED, "egobox is not installed")
    def test_branin(self):
        self.pb = pb = om.Problem(BraninMDA())
        pb.model.add_design_var("x1", lower=-5, upper=10)
        pb.model.add_design_var("x2", lower=0, upper=15)
        pb.model.add_objective("obj")
        pb.model.add_constraint("con", upper=0)
        case_recorder_filename = "test_egobox_driver_branin.sqlite"
        self._check_recorder_file(pb, cstr=True, filename=case_recorder_filename)

    @unittest.skipIf(EGOBOX_NOT_INSTALLED, "egobox is not installed")
    def test_ackley(self):
        self.pb = pb = om.Problem(AckleyMDA())
        pb.model.add_design_var("x", lower=-32.768, upper=32.768)
        pb.model.add_objective("obj")
        case_recorder_filename = "test_egobox_driver_ackley.sqlite"
        self._check_recorder_file(pb, cstr=False, filename=case_recorder_filename)

    def _check_recorder_file(self, pb, cstr, filename):
        pb.driver = EgoboxEgorDriver()
        pb.driver.options["optimizer"] = "EGOR"
        pb.driver.opt_settings["maxiter"] = 10
        pb.setup()
        self.case_recorder_filename = "{}/{}".format(pb.get_outputs_dir(), filename)
        recorder = om.SqliteRecorder(self.case_recorder_filename)
        pb.model.add_recorder(recorder)
        self.pb.run_driver()
        self.assertTrue(os.path.exists(self.case_recorder_filename))
        reader = om.CaseReader(self.case_recorder_filename)
        for case_id in reader.list_cases():
            case = reader.get_case(case_id)
            print(case.outputs["obj"])


if __name__ == "__main__":
    unittest.main()
