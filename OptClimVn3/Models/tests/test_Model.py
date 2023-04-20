import copy
import logging
import pathlib
import shutil
import tempfile
import time
import unittest
import unittest.mock
import filecmp
import datetime
import subprocess
import os

import f90nml
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

import genericLib
from Models.Model import Model
from Models.namelist_var import namelist_var
from Models.Model import register_param


# To get log info set --log-cli-level WARNING in "additional arguments " in pycharm config

class myModel(Model):
    @register_param('RHCRIT')
    def cloudRHcrit(self, rhcrit, inverse=False):
        """
        Compute rhcrit on multiple model levels
        :param rhcrit: meta parameter for rhcrit
        :param inverse: default False. If True invert the relationship
        :return: (value of meta parameter if inverse set otherwise
           a tuple with namelist_var infor and  a list of rh_crit on model levels

        """
        # Check have 19 levels.
        rhcrit_nl = namelist_var(filepath=pathlib.Path('CNTLATM'), nl_var='RHCRIT', namelist='RUNCNST')
        curr_rhcrit = rhcrit_nl.read_value(dirpath=self.model_dir)
        if len(curr_rhcrit) != 19:
            raise ValueError("Expect 19 levels")

        if inverse:
            return rhcrit_nl.read_value(dirpath=self.model_dir)[3]
        else:
            cloud_rh_crit = 19 * [rhcrit]
            cloud_rh_crit[0] = max(0.95, rhcrit)
            cloud_rh_crit[1] = max(0.9, rhcrit)
            cloud_rh_crit[2] = max(0.85, rhcrit)
            return (rhcrit_nl, cloud_rh_crit)

    print('hello')


myModel.update_from_file(myModel.expand("$OPTCLIMTOP/OptClimVn3/Models/tests/example_Parameters.csv"), duplicate=False)


class ModelTestCase(unittest.TestCase):
    def fake_fn(self):
        param = self.model.parameters  # get the parameters
        sim_obs = dict()
        obs_count = 0
        for k, v in param.items():
            oname = f"obs{obs_count}"
            sim_obs[oname] = v ** 2
            obs_count += 1
        return pd.Series(sim_obs)

    def setUp(self) -> None:
        """
        Setup!
        :return:
        """

        # test faking!

        # logger = logging.getLogger(__name__)
        logging.basicConfig(format='%(asctime)s %(module)s   %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
        # create a model and store it.
        tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(tmpDir.name)  # used throughout.
        refDir = pathlib.Path(Model.expand('$OPTCLIMTOP/Configurations')) / 'xnmea'  # need a coupled model.
        post_process = dict(script='$OPTCLIMTOP/OptClimVn2/comp_obs.py', outputPath='obs.json')
        self.model = myModel(name='test_model', reference=refDir,
                             model_dir=testDir, post_process=post_process,
                             parameters=dict(RHCRIT=2, VF1=2.5))
        self.tmpDir = tmpDir
        self.testDir = testDir  # for clean up!
        self.refDir = refDir
        self.config_path = self.model.config_path

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        # self.tmpDir.cleanup() # sadly fails because not all files in are writable.
        genericLib.delDirContents(self.testDir)

    def assertAllequal(self, data1, data2):
        for key, value in data1.items():
            value2 = data2[key]
            if isinstance(value, pd.Series):
                pdtest.assert_series_equal(value, value2)
            elif isinstance(value, pd.DataFrame):
                pdtest.assert_frame_equal(value, value2)
            elif isinstance(value, np.ndarray):
                nptest.assert_equal(value, value2)
            else:
                self.assertEqual(value, value2)

    def tearDown(self):
        shutil.rmtree(self.testDir, onerror=genericLib.errorRemoveReadonly)

    def test_inherit(self):
        """
        Test that class inheritance and naming works
        :return:
        """

        # define a bunch of sub-classes to check all works

        class model1(Model):

            def __init__(self, *args, **kwargs):
                if kwargs.get("verbose", False):
                    print("I am a ", self.class_name(), args, kwargs)
                self.values = copy.deepcopy(kwargs)
                self.Fred = kwargs.get('Fred', 23)

            def __repr__(self):
                return str(f"{self.class_name()} values: {self.values} Fred {self.Fred} ")

        class model2(model1):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.Harry = kwargs.get('Harry', 1901)

            def __repr__(self):
                s = super().__repr__()
                return f"{s} Harry: {self.Harry}"

        class model2mod(model1):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.James = kwargs.get('James', 1902)

            def __repr__(self):
                s = super().__repr__()
                return f"{s} James: {self.James}"

        class model3(model2):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.Simon = kwargs.get('Simon', 1965)

            def __repr__(self):
                s = super().__repr__()
                return f"{s} Simon: {self.Simon}"

        class model4(model3):
            pass

        # test things.
        self.assertEqual(['Model', 'myModel', 'model1', 'model2', 'model2mod', 'model3', 'model4'],
                         Model.known_models(), )
        oh = 2
        init1 = dict(name='xdr56',
                     post_process=dict(high='five', script='$OPTCLIMTOP/OptClimVn2/comp_obs.py', outputPath='obs.json'),
                     parameters=dict(harry=2, fred=oh))
        init2 = copy.copy(init1)
        init2['post_process']['high'] = 'four'
        init2['parameters'] = dict(harry=2, fredSmith=2, fred=oh)
        for name in ['Model', 'model1', 'model2', 'model3', 'model2mod', 'model4']:
            t = Model.model_init(name, **init1)
            t2 = Model.model_init(name, **init2)
            self.assertTrue(type(t) == type(t2), 'Types differ')
            self.assertEqual(t.class_name(), type(t).__name__)
        for name in ['unknown', 'model3aa']:  # unknown models should raise exceptions
            with self.assertRaises(ValueError):
                t = Model.model_init(name, **init1)

    def test_add_param_info(self):
        nl1 = namelist_var(filepath=pathlib.Path('fred'), namelist='james', nl_var='harry')
        nl2 = namelist_var(filepath=pathlib.Path('fred'), namelist='james', nl_var='james')

        expect_param_info = [nl1, nl2]
        pardict = dict(fred=2, james=3)
        model = Model.model_init('myModel', name='test_model', reference=self.refDir,
                                 model_dir=self.testDir, parameters=pardict)
        model.add_param_info(dict(vf1=nl1))
        model.add_param_info(dict(vf1=nl2), duplicate=True)
        self.assertEqual(model.param_info.to_dict()['vf1'], expect_param_info)
        # should expect an error now
        with self.assertRaises(ValueError):
            model.add_param_info(dict(entcoef=nl2))

    def test_to_dict(self):
        """
        Test that conversion to a dict works. Which calls super class,
        :return:
        """
        pardict = dict(fred=2, james=3)
        model = Model(name='test_model', reference=self.refDir,
                      model_dir=self.testDir, parameters=pardict)
        expected_dct = dict(name='test_model', reference=self.refDir,
                            model_dir=self.testDir, parameters=pardict,
                            submit_script='submit.sh', continue_script='continue.sh',
                            post_process={}, post_process_cmd=None, output={},
                            post_process_output=None,
                            post_process_script=None, post_process_script_interp=None, fake=False, simulated_obs=None,
                            perturb_count=0,config_path=self.testDir/"test_model.mcfg",
                            status='CREATED', history=model.history)
        dct = model.to_dict()
        self.assertEqual(expected_dct, dct)

    def test_load_dump(self):
        """
        Test we can dump a Model and that loading it gives us the same thing.
        Dict contains what we expect!
        :return:
        """
        pardict = dict(fred=2, james=3)

        class model4(Model):
            def fred(self, message):
                print(f"Fred says: {message}")

        class model5(model4):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fredv = 10

        for class_name in ['Model', 'model4', 'model5']:
            model = Model.model_init(class_name, name=f'test_{class_name}01', reference=self.refDir,
                                     model_dir=self.testDir, parameters=pardict)
            model.dump_model()
            lmodel = Model.load_model(model.config_path)
            self.assertEqual(lmodel.class_name(), class_name)
            self.assertAllequal(vars(model), vars(lmodel))
        lmodel.fred("Hello")
        self.assertEqual(lmodel.fredv, 10)

    def test_gen_params(self):
        # test gen params works
        shutil.copytree(self.refDir, self.tmpDir.name, symlinks=True, dirs_exist_ok=True)
        nl_iter = self.model.gen_params()
        # work out what we expect...
        expect = []
        for nl, value in self.model.param_info.gen_parameters(self.model, **self.model.parameters):
            expect.append((nl, value))
        self.assertEqual(nl_iter, expect)

    def test_set_params(self):
        # test setting params works.
        shutil.copytree(self.refDir, self.model.model_dir, symlinks=True, dirs_exist_ok=True)

        self.model.set_params()
        # use gen_params to get parameters aand then check they are as expected.
        nl_iter = self.model.gen_params()
        for (nl, value) in nl_iter:
            got = nl.read_value(dirpath=self.model.model_dir, clean=True)
            self.assertEqual(value, got)

    def test_changed_nl(self):
        shutil.copytree(self.refDir, self.model.model_dir, symlinks=True, dirs_exist_ok=True)
        change_nl = self.model.changed_nl()
        nl_iter = self.model.gen_params()
        expect_nl = dict()
        for (nl, value) in nl_iter:
            filepath = self.model.model_dir / nl.filepath
            if filepath not in expect_nl.keys():
                expect_nl[filepath] = f90nml.read(filepath)  # read everything!
            # overwrite the changed values.
            expect_nl[filepath][nl.namelist][nl.nl_var] = value
        for file, namelist in change_nl.items():
            for key, nlc in namelist.items():  # iterate over changed namelists
                self.assertEqual(nlc, expect_nl[filepath][key])

    def test_create_model(self):
        """
        Test that create model works.
        create_model copies reference directory to model_dir
        So we check that reference_dir and model_dir are identical
        :return:
        """

        self.model.create_model()
        # verify that reference and model_dir are identical
        self.assertTrue(filecmp.dircmp(self.model.reference, self.model.model_dir))

    def test_set_status(self):
        """
        Test set_status works
        :return:
        """
        # test the expected path works
        nhist = len(self.model.history)
        for status in ['INSTANTIATED', 'SUBMITTED', 'RUNNING', "FAILED", "PERTURBED", "SUBMITTED", "RUNNING",
                       'SUCCEEDED', 'PROCESSED']:
            time.sleep(1e-3)  # sleep for a millisecond
            omodel = copy.deepcopy(self.model)
            with self.assertLogs(level=logging.DEBUG) as log:
                self.model.set_status(status)
            # only check first log entry which is the status change.
            self.assertEqual(log.output[0],
                             f"DEBUG:root:Changing status from {omodel.status} to {self.model.status}")
            # check status is as expected
            self.assertEqual(self.model.status, status)
            nhist += 1  # 1 more history entry
            # verify all but status and history are the same form model prior to status change,
            omodeld = vars(omodel)
            modeld = vars(self.model)
            keys_to_check = set(omodeld.keys()) - {'history', 'status'}
            for key in keys_to_check:
                self.assertEqual(modeld[key], omodeld[key])
            for key in ['history', 'status']:  # should be different
                self.assertNotEqual(modeld[key], omodeld[key])
            # history only differs in last entry
            h = copy.deepcopy(modeld['history'])
            h.popitem()
            self.assertEqual(h, omodeld['history'])

            # read in the model
            lmodel = Model.load_model(self.config_path)
            self.assertEqual(vars(lmodel), vars(self.model))  # check they are the same
            self.assertEqual(nhist, len(self.model.history))  # history right length
            # test failures

            with self.assertRaises(ValueError):
                self.model.set_status("CREATED")
            with self.assertRaises(ValueError):
                self.model.set_status("FLARTIBARTFAST")

    def test_update_history(self):
        """ Verify that update_history works.
        Two cases to consider
            1) Multiple updates in short time. Need to "mock" datetime.datetime.now(tz=datetime.timezone.utc)
            2) Updates at different times.

         """

        expected = dict()
        model = self.model
        model.history = {}  # make history empty to start with
        test_now = datetime.datetime(2023, 4, 16, 17, 24, 50)
        with unittest.mock.patch('datetime.datetime', wraps=datetime.datetime) as dt:
            dt.now.return_value = test_now
            for count in range(0, 20):
                msg = f"Count is {count}"
                now = str(dt.now(tz=datetime.timezone.utc))
                lst = expected.get(now, [])
                lst.append(msg)
                model.update_history(msg)
                expected[now] = lst
            # now update times at 1 second difference
            for count in range(0, 20):
                test_now += datetime.timedelta(seconds=1)
                dt.now.return_value = test_now
                msg = f"Count is {count}"
                now = str(dt.now(tz=datetime.timezone.utc))
                model.update_history(msg)
                expected[now] = [msg]

        self.assertEqual(model.history, expected)

    def test_read_values(self):
        """
        Test can read values.
        :return:
        """
        model = copy.deepcopy(self.model)
        model.model_dir = model.reference
        # expect values to as defined!
        expect_dir = dict(VF1=1, RHCRIT=0.7, ENTCOEF=3.0, G0=10)  # some params including a function.
        got = model.read_values(list(expect_dir.keys()))
        self.assertEqual(expect_dir, got)

    def test_instantiate(self):
        """
        Test instantiate method.
          Files should exist.  There should be .bak files for those namelists that got changed.
          These .bak files should be identical to reference files.
          Model status should be instantiated
          Assume that detailed tests on changes to files already done.

        :return:
        """
        # model_dir should only have one file.

        omodel = copy.deepcopy(self.model)
        self.model.instantiate()

        mm = Model.load_model(self.config_path)
        self.assertEqual(vars(mm), vars(self.model))
        dd = vars(self.model)
        dd2 = vars(omodel)
        dd2['status'] = 'INSTANTIATED'
        self.assertNotEqual(dd.pop('history'), dd2.pop('history'))
        self.assertEqual(dd, dd2)
        count_config = 0
        bak_count = 0
        # how many .bak files do we expect?
        lst = self.model.gen_params()  # lst is tuples of namelist, value. (might need to generalise this test in future)
        nl_changed_files = dict()
        for (nl, value) in lst:
            nl_changed_files[self.model.model_dir / nl.filepath] = True

        expected_bak_count = len(nl_changed_files)

        for file in self.model.model_dir.iterdir():
            # if file ends in .bak then compare it with equiv file from ref dir
            if file.is_dir():
                raise ValueError("Got a dir.")  # should not have dir but cannot be bothered checking sub-dirs etc!
            if file == self.model.config_path:  # it's a model config
                count_config += 1
                continue
            if file.suffix == '.bak':
                ref_file = self.model.reference / (file.stem)
                self.assertTrue(filecmp.cmp(file, ref_file))
                bak_count += 1
                self.assertTrue(nl_changed_files[file.parent / file.stem])
            else:
                # see if we have a .bak file.
                bak_file = file.parent / (file.name + ".bak")
                if bak_file.exists():  # got it -- skip
                    continue
                else:
                    self.assertTrue(filecmp.cmp(file, self.model.reference / file.name))

        self.assertEqual(bak_count, expected_bak_count)
        self.assertEqual(1, count_config)  # only one config file.

    def test_run_model(self):
        """
        Test run_model works.
         Test status outside expected fails
        :return:
        """

        def submit_cmd(x):
            """
            Dummy submit_cmd
            :param x: some variable
            :return: str(x)
            """
            return x

        self.model.status = 'INSTANTIATED'  # should have state instantiated
        model = self.model
        omodel = copy.deepcopy(model)
        time.sleep(0.01)  # little sleep so history generates a new entry
        result = model.run_model(submit_cmd, post_process_cmd=None)
        # expect result to be str(self.submit_script)
        self.assertEqual(result, submit_cmd(model.submit_script))
        # also expect changes in status, history, and post_process_cmd
        expect_history = omodel.history
        model.history.popitem()  # remove last history entry.
        self.assertEqual(expect_history, model.history)
        self.assertEqual(model.status, 'SUBMITTED')
        self.assertEqual(model.post_process_cmd, None)
        self.assertEqual(model.output['SUBMITTED'], [submit_cmd(model.submit_script)])
        # test Continued
        model = copy.deepcopy(omodel)
        result = model.run_model(submit_cmd, post_process_cmd='some cmd', new=False)
        # expect result to be str(self.submit_script)
        self.assertEqual(result, submit_cmd(model.continue_script))
        # also expect changes in status, history, and post_process_cmd
        expect_history = omodel.history
        model.history.popitem()  # remove last history entry.
        self.assertEqual(expect_history, model.history)
        self.assertEqual(model.status, 'SUBMITTED')
        self.assertEqual(model.post_process_cmd, 'some cmd')
        self.assertEqual(model.output['SUBMITTED'], [submit_cmd(model.continue_script)])
        self.assertFalse(model.fake)

        model = self.model
        model.status = 'INSTANTIATED'
        model.run_model(submit_cmd, fake_function=self.fake_fn)
        self.assertEqual(model.status, 'PROCESSED')
        self.assertIsNone(model.post_process_cmd)
        expect = self.fake_fn().rename(model.name)
        pdtest.assert_series_equal(model.simulated_obs, expect)

    def test_running(self):
        """
        Test running.
        Changes after wards is expect a file.
        :return:
        """

        model = copy.deepcopy(self.model)
        model.status = 'SUBMITTED'
        test_now = datetime.datetime(2023, 4, 16, 17, 24, 50)
        with unittest.mock.patch('datetime.datetime', wraps=datetime.datetime) as dt:
            test_now += datetime.timedelta(seconds=1)
            dt.now.return_value = test_now
            model.running()
        self.assertEqual(model.status, 'RUNNING')
        dmodel = Model.load_model(self.config_path)
        self.assertEqual(vars(dmodel), vars(model))
        h = model.history
        h.popitem()
        self.assertEqual(self.model.history, h)

    def test_perturb(self):
        """
        Pertub expects a known set of parameters to be passed in.
        Will update parameters and then set them.
        Expect parameters to be changed and a bunch of files changed.
        Status should be PERTURBED. (Then it can be submitted).
        :return:
        """
        model = self.model

        model.instantiate()  # now instantiated
        model.status = 'FAILED'
        v = model.read_values('G0')
        v['G0'] *= (1 + 1e-7)  # small perturb
        with self.assertLogs(level='DEBUG') as log:
            test_now = datetime.datetime(2023, 4, 16, 17, 24, 50)
            with unittest.mock.patch('datetime.datetime', wraps=datetime.datetime) as dt:
                dt.now.return_value = test_now
                model.perturb(v)
        self.assertEqual(log.output[-1], f"DEBUG:root:set parameters to {v}")
        self.assertEqual(len(model.history),
                         3)  # expect 3 bits of history. Created, Instantiated, the  perturbed using and setting status
        p = model.read_values('G0')
        self.assertEqual(p, v)
        self.assertEqual(model.perturb_count, 1)  # perturbed it once.
        self.assertEqual(model.status, 'PERTURBED')

    def test_failed(self):
        """
        Should set status to FAILED.
        :return:
        """

        model = copy.deepcopy(self.model)
        model.status = 'RUNNING'
        test_now = datetime.datetime(2023, 4, 16, 17, 24, 50)
        with unittest.mock.patch('datetime.datetime', wraps=datetime.datetime) as dt:
            test_now += datetime.timedelta(seconds=1)
            dt.now.return_value = test_now
            model.failed()
        self.assertEqual(model.status, 'FAILED')
        dmodel = Model.load_model(self.config_path)
        self.assertEqual(vars(dmodel), vars(model))
        h = model.history
        h.popitem()
        self.assertEqual(self.model.history, h)

    def test_succeeded(self):
        """
        Test that succeeded worked!
        :return:
        """
        model = copy.deepcopy(self.model)
        model.status = 'RUNNING'  # should be RUNNING
        test_now = datetime.datetime(2023, 4, 16, 17, 24, 50)
        with unittest.mock.patch('datetime.datetime', wraps=datetime.datetime) as dt:
            test_now += datetime.timedelta(seconds=1)
            dt.now.return_value = test_now
            with self.assertLogs() as log:
                model.succeeded()
            self.assertEqual(log.output[0], 'INFO:root:No post-processing processing cmd')
        self.assertEqual(model.status, 'SUCCEEDED')
        dmodel = Model.load_model(self.config_path)
        self.assertEqual(vars(dmodel), vars(model))
        h = model.history
        h.popitem()
        self.assertEqual(self.model.history, h)
        self.assertEqual(model.output['SUCCEEDED'],[None])
        # run the model with a bad script
        model.status = 'RUNNING'
        model.post_process_cmd = 'noscript.sh'
        with self.assertRaises(subprocess.CalledProcessError):
            model.succeeded()

    def test_process(self):
        """
        Test process works.
        Will do first by faking things!
        :return:
        """
        model = copy.deepcopy(self.model)
        model.fake = True
        model.status = 'SUCCEEDED'  # we have succeeded
        model.process()  # with fake
        self.assertEqual(model.status, 'PROCESSED')
        # no fake fn..
        model.fake = False
        model.status = 'SUCCEEDED'  # we have succeeded
        # use fake_fn to generate some fake obs!
        fake_obs = self.fake_fn()
        model.post_process['fake_obs'] = fake_obs.to_dict()
        model.post_process_script = model.expand('$OPTCLIMTOP/OptClimVn3/Models/scripts/pp_script_test.py')

        if os.name == 'nt':  # running on Windows.  Test script is a python script!
            model.post_process_script_interp = 'python'
        model.post_process_json = 'pp.json'
        model.post_process_output = 'obs.json'
        model.process()
        self.assertEqual(model.status, 'PROCESSED')

    def test_end_to_end(self):
        """
        Test that end to end case works.
        :return:
        """
        submit_fn = lambda x: str(x)
        def sleep_load(cfg):

            wait_time = 0.01#
            time.sleep(wait_time)
            model = Model.load_model(cfg)
            return model
        post_process = dict(script='$OPTCLIMTOP/OptClimVn3/Models/scripts/pp_script_test.py',
                            outputPath='obs.json',
                            fake_obs=self.fake_fn().to_dict())
        if os.name == 'nt':  # running on Windows. Test script is a python script!
            post_process.update(script_interp='python')
        cfg = self.testDir/'model0001.mcfg'
        model = myModel(name='test_model00', reference=self.refDir, model_dir=self.testDir,
                        config_path=cfg,
                        parameters=dict(VF1=1, RHCRIT=0.7, G0=11),
                        post_process=post_process)  # create the model.
        time.sleep(0.01)
        model.instantiate()  # instantiate the model.
        model = sleep_load(cfg)

        model.run_model(submit_fn)  # submit the model.
        model = sleep_load(cfg)

        model.running()
        model = sleep_load(cfg)

        model.succeeded()  # model has succeeded
        model = sleep_load(cfg)

        model.process()  # and do the post-processing
        # need to run a bunch of tests here.
        # having got to here should have simulated_obs be post_process['fake_obs']
        pdtest.assert_series_equal(model.simulated_obs,pd.Series(post_process['fake_obs']).rename(model.name))
        # should have 6 history entries.
        self.assertEqual(len(model.history),6)
        # three  outputs -- from submission, SUCCEEDED (which submited the pp job if requested) and post-processing
        self.assertEqual(len(model.output),3)


        # now do but where model fails, get perturbed, gets continued and then works.
        time.sleep(0.01)
        cfg = self.testDir / 'model0002.mcfg'
        model = myModel(name='test_model02', reference=self.refDir, model_dir=self.testDir,
                        config_path=cfg,
                        parameters=dict(VF1=1, RHCRIT=0.7, G0=11),
                        post_process=post_process)  # create the model.
        time.sleep(0.01)
        model.instantiate()  # instantiate the model.
        model = sleep_load(cfg)

        model.run_model(submit_fn)  # submit the model.
        model = sleep_load(cfg)

        model.running()
        model = sleep_load(cfg)

        model.failed()
        model = sleep_load(cfg)

        ct = model.read_values('CT')
        print(ct)
        ct['CT'] *= (1+1e-6)
        model.perturb(ct)
        model = sleep_load(cfg)

        model.run_model(submit_fn,new=False) # continue
        model = sleep_load(cfg)

        model.running()
        model = sleep_load(cfg)

        model.succeeded()  # model has succeeded
        model = sleep_load(cfg)

        model.process()  # and do the post-processing

        # should have 10 history entries.
        self.assertEqual(len(model.history),10)
        # three  outputs -- from submissions, SUCCEEDED (which submitted the pp job if requested) and post-processing
        self.assertEqual(len(model.output),3)
        # submissions should have two entries.
        expect_lens=dict(SUBMITTED=2,SUCCEEDED=1,PROCESSED=1)
        for k,v in expect_lens.items():
            self.assertEqual(len(model.output[k]),v,msg=f"Failed for {k} expected {v}")




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
