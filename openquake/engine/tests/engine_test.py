# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2010-2016 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import sys
import getpass
import subprocess
import tempfile
import unittest
from StringIO import StringIO
import mock

from openquake.server.db import models
from django.core import exceptions

from openquake.engine import engine
from openquake.engine.tests.utils import helpers


class FakeJob(object):
    def __init__(self, job_type, calculation_mode):
        self.job_type = job_type
        self.calculation_mode = calculation_mode


class CheckHazardRiskConsistencyTestCase(unittest.TestCase):
    def test_ok(self):
        haz_job = FakeJob('hazard', 'scenario')
        engine.check_hazard_risk_consistency(
            haz_job, 'scenario_risk')

    def test_obsolete_mode(self):
        haz_job = FakeJob('hazard', 'scenario')
        with self.assertRaises(ValueError) as ctx:
            engine.check_hazard_risk_consistency(
                haz_job, 'scenario')
        msg = str(ctx.exception)
        self.assertEqual(msg, 'Please change calculation_mode=scenario into '
                         'scenario_risk in the .ini file')

    def test_inconsistent_mode(self):
        haz_job = FakeJob('hazard', 'scenario')
        with self.assertRaises(engine.InvalidCalculationID) as ctx:
            engine.check_hazard_risk_consistency(
                haz_job, 'classical_risk')
        msg = str(ctx.exception)
        self.assertEqual(
            msg, "In order to run a risk calculation of kind "
            "'classical_risk', you need to provide a "
            "calculation of kind ['classical', 'classical_risk'], "
            "but you provided a 'scenario' instead")


class JobFromFileTestCase(unittest.TestCase):

    def test_create_job_default_user(self):
        job = engine.create_job()

        self.assertEqual('openquake', job.user_name)
        self.assertEqual('pre_executing', job.status)

        # Check the make sure it's in the database.
        try:
            models.OqJob.objects.get(id=job.id)
        except exceptions.ObjectDoesNotExist:
            self.fail('Job was not found in the database')

    def test_create_job_specified_user(self):
        user_name = helpers.random_string()
        job = engine.create_job(user_name=user_name)

        self.assertEqual(user_name, job.user_name)
        self.assertEqual('pre_executing', job.status)

        try:
            models.OqJob.objects.get(id=job.id)
        except exceptions.ObjectDoesNotExist:
            self.fail('Job was not found in the database')


class RunCalcTestCase(unittest.TestCase):
    """
    Test engine.run_calc in case of errors
    """
    def test(self):
        cfg = helpers.get_data_path('event_based_hazard/job.ini')
        job = engine.job_from_file(cfg, 'test_user')
        with tempfile.NamedTemporaryFile() as temp:
            with self.assertRaises(ZeroDivisionError), mock.patch(
                    'openquake.engine.engine._do_run_calc', lambda *args: 1/0
            ), mock.patch('openquake.engine.engine.cleanup_after_job',
                          lambda job: None):
                engine.run_calc(job, 'info', temp.name, exports=[])
            logged = open(temp.name).read()

            # make sure the real error has been logged
            self.assertIn('integer division or modulo by zero', logged)


class OpenquakeCliTestCase(unittest.TestCase):
    """
    Run "oq-engine --version" as a separate process using `subprocess`.
    """

    def test_run_version(self):
        args = [helpers.RUNNER, "--version"]

        print 'Running:', ' '.join(args)  # this is useful for debugging
        return subprocess.check_call(args)


class DeleteHazCalcTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.hazard_cfg = helpers.get_data_path(
            'simple_fault_demo_hazard/job.ini')
        cls.risk_cfg = helpers.get_data_path(
            'classical_psha_based_risk/job.ini')

    def test_del_calc(self):
        raise unittest.SkipTest
        hazard_job = helpers.get_job(
            self.hazard_cfg, username=getpass.getuser())

        models.Output.objects.create_output(
            hazard_job, 'test_curves_1', output_type='hazard_curve',
            ds_key='hcurve'
        )
        models.Output.objects.create_output(
            hazard_job, 'test_curves_2', output_type='hazard_curve',
            ds_key='hcurve'
        )

        # Sanity check: make sure the hazard calculation and outputs exist in
        # the database:
        hazard_jobs = models.OqJob.objects.filter(id=hazard_job.id)
        self.assertEqual(1, hazard_jobs.count())

        outputs = models.Output.objects.filter(oq_job=hazard_job.id)
        self.assertEqual(2, outputs.count())

        # Delete the calculation
        engine.del_calc(hazard_job.id)

        # Check that the hazard calculation and its outputs were deleted:
        outputs = models.Output.objects.filter(oq_job=hazard_job.id)
        self.assertEqual(0, outputs.count())

        hazard_jobs = models.OqJob.objects.filter(id=hazard_job.id)
        self.assertEqual(0, hazard_jobs.count())

    def test_del_calc_does_not_exist(self):
        self.assertRaises(RuntimeError, engine.del_calc, -1)

    def test_del_calc_no_access(self):
        # Test the case where we try to delete a hazard calculation which does
        # not belong to current user.
        # In this case, deletion is now allowed and should raise an exception.
        hazard_job = helpers.get_job(
            self.hazard_cfg, username=helpers.random_string())
        self.assertRaises(RuntimeError, engine.del_calc, hazard_job.id)

    def test_del_calc_referenced_by_risk_calc(self):
        raise unittest.SkipTest
        # Test the case where a risk calculation is referencing the hazard
        # calculation we want to delete.
        # In this case, deletion is not allowed and should raise an exception.
        risk_job, _ = helpers.get_fake_risk_job(
            self.risk_cfg, self.hazard_cfg,
            output_type='curve', username=getpass.getuser()
        )
        hc = risk_job.hazard_calculation
        self.assertRaises(RuntimeError, engine.del_calc, hc.id)

    def test_del_calc_output_referenced_by_risk_calc(self):
        raise unittest.SkipTest
        # Test the case where a risk calculation is referencing one of the
        # belonging to the hazard calculation we want to delete.
        # In this case, deletion is not allowed and should raise an exception.
        risk_job, _ = helpers.get_fake_risk_job(
            self.risk_cfg, self.hazard_cfg,
            output_type='curve', username=getpass.getuser()
        )
        hc = risk_job.hazard_calculation
        self.assertRaises(RuntimeError, engine.del_calc, hc.id)


class DeleteRiskCalcTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.hazard_cfg = helpers.get_data_path(
            'simple_fault_demo_hazard/job.ini')
        cls.risk_cfg = helpers.get_data_path(
            'classical_psha_based_risk/job.ini')

    def test_del_calc(self):
        raise unittest.SkipTest
        risk_job, _ = helpers.get_fake_risk_job(
            self.risk_cfg, self.hazard_cfg,
            output_type='curve', username=getpass.getuser()
        )
        models.Output.objects.create_output(
            risk_job, 'test_curves_1', output_type='loss_curve',
            ds_key='rcurves-rlzs'
        )
        models.Output.objects.create_output(
            risk_job, 'test_curves_2', output_type='loss_curve',
            ds_key='rcurves-rlzs'
        )

        # Sanity check: make sure the risk calculation and outputs exist in
        # the database:
        risk_calcs = models.OqJob.objects.filter(id=risk_job.id)
        self.assertEqual(1, risk_calcs.count())

        outputs = models.Output.objects.filter(oq_job=risk_job.id)
        self.assertEqual(2, outputs.count())

        # Delete the calculation
        engine.del_calc(risk_job.id)

        # Check that the risk calculation and its outputs were deleted:
        outputs = models.Output.objects.filter(oq_job=risk_job.id)
        self.assertEqual(0, outputs.count())

        risk_calcs = models.OqJob.objects.filter(id=risk_job.id)
        self.assertEqual(0, risk_calcs.count())

    def test_del_calc_does_not_exist(self):
        self.assertRaises(RuntimeError, engine.del_calc, -1)

    def test_del_calc_no_access(self):
        raise unittest.SkipTest
        # Test the case where we try to delete a risk calculation which does
        # not belong to current user.
        # In this case, deletion is now allowed and should raise an exception.
        risk_job, _ = helpers.get_fake_risk_job(
            self.risk_cfg, self.hazard_cfg,
            output_type='curve', username=helpers.random_string()
        )
        self.assertRaises(RuntimeError, engine.del_calc, risk_job.id)


class FakeOutput(object):
    def __init__(self, id, output_type, display_name):
        self.id = id
        self.output_type = output_type
        self.display_name = display_name

    def get_output_type_display(self):
        return self.display_name + str(self.id)


class PrintSummaryTestCase(unittest.TestCase):
    outputs = [FakeOutput(i, 'gmf', 'gmf') for i in range(1, 12)]

    def print_outputs_summary(self, full):
        orig_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            engine.print_outputs_summary(self.outputs, full)
            got = sys.stdout.getvalue()
        finally:
            sys.stdout = orig_stdout
        return got

    def test_print_outputs_summary_full(self):
        self.assertEqual(self.print_outputs_summary(full=True), '''\
  id | output_type | name
   1 | gmf1 | gmf
   2 | gmf2 | gmf
   3 | gmf3 | gmf
   4 | gmf4 | gmf
   5 | gmf5 | gmf
   6 | gmf6 | gmf
   7 | gmf7 | gmf
   8 | gmf8 | gmf
   9 | gmf9 | gmf
  10 | gmf10 | gmf
  11 | gmf11 | gmf
''')

    def test_print_outputs_summary_short(self):
        self.assertEqual(
            self.print_outputs_summary(full=False), '''\
  id | output_type | name
   1 | gmf1 | gmf
   2 | gmf2 | gmf
   3 | gmf3 | gmf
   4 | gmf4 | gmf
   5 | gmf5 | gmf
   6 | gmf6 | gmf
   7 | gmf7 | gmf
   8 | gmf8 | gmf
   9 | gmf9 | gmf
  10 | gmf10 | gmf
 ... | gmf11 | 1 additional output(s)
Some outputs where not shown. You can see the full list with the command
`oq-engine --list-outputs`
''')
