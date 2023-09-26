import copy
import datetime
import filecmp
import importlib.resources
import logging
import os
import pathlib
import shutil
import tempfile
import time
import unittest
import unittest.mock


import StudyConfig # so can read in a config for fake_fn.
import f90nml
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
import engine
import genericLib
import generic_json
from Models import Model
from Model import register_param
from namelist_var import namelist_var

#++++++++++++++liangwj++++++++++++++
from contextlib import contextmanager
import tempfile
import shutil
@contextmanager
def open_as_file(path):
	temp_dir = tempfile.mkdtemp()
	temp_path = os.path.join(temp_dir, os.path.basename(path))

	shutil.copy2(path, temp_path)

	try:
		yield temp_path
	finally:
		shutil.rmtree(temp_dir)
#++++++++++++++liangwj++++++++++++++



def gen_time():
    # used to mock Model.now()
    time = datetime.datetime(2000, 1, 11, 0, 0, 0)
    timedelta = datetime.timedelta(seconds=1)
    while True:
        time += timedelta
        yield time


# To get log info set --log-cli-level WARNING in "additional arguments " in pycharm config
# remove any registered classes **except** Model.
for k in Model.known_models():
    if k != "Model":
        Model.remove_class(k)


class myModel(Model):
    @register_param('RHCRIT')
    def cloudRHcrit(self, rhcrit):
        """
        Compute rhcrit on multiple model levels
        :param rhcrit: meta parameter for rhcrit. If Noen relationship will be inverted.
        :return: (value of meta parameter if inverse set otherwise
           a tuple with namelist_var infor and  a list of rh_crit on model levels

        """
        # Check have 19 levels.
        rhcrit_nl = namelist_var(filepath=pathlib.Path('CNTLATM'), nl_var='RHCRIT', namelist='RUNCNST')
        curr_rhcrit = rhcrit_nl.read_value(dirpath=self.model_dir)
        if len(curr_rhcrit) != 19:
            raise ValueError("Expect 19 levels")
        inverse = rhcrit is None
        if inverse:
            return rhcrit_nl.read_value(dirpath=self.model_dir)[3]
        else:
            cloud_rh_crit = 19 * [rhcrit]
            cloud_rh_crit[0] = max(0.95, rhcrit)
            cloud_rh_crit[1] = max(0.9, rhcrit)
            cloud_rh_crit[2] = max(0.85, rhcrit)
            return rhcrit_nl, cloud_rh_crit

    print('hello')


# myModel.update_from_file(myModel.expand("$OPTCLIMTOP/OptClimVn3/Models/tests/example_Parameters.csv"), duplicate=False)

#++++++++++++liangwj+++++++++++
path_to_file = "/BIGDATA2/sysu_atmos_wjliang_1/FG3/newModelOptimisation/OptClimVn3/Models/parameter_config/example_Parameters.csv"
with open_as_file(path_to_file) as pth:
    myModel.update_from_file(pth)
root_pth  = "/BIGDATA2/sysu_atmos_wjliang_1/FG3/newModelOptimisation/OptClimVn3"
config = StudyConfig.readConfig("/BIGDATA2/sysu_atmos_wjliang_1/FG3/newModelOptimisation/OptClimVn3/configurations/dfols14param_opt3.json")
#++++++++++++liangwj+++++++++++

# traverse = importlib.resources.files("Models")
# with importlib.resources.as_file(traverse.joinpath("parameter_config/example_Parameters.csv")) as pth:
#     myModel.update_from_file(pth)
#
# root_pth  = importlib.resources.files("OptClimVn3")
# config = StudyConfig.readConfig(root_pth/"configurations/dfols14param_opt3.json")

def fake_function(param):
    return genericLib.fake_fn(config,param)



class ModelTestCase(unittest.TestCase):

    def fake_fn(self):
        return fake_function(self.model.parameters)

    def setUp(self) -> None:
        """
        Setup!
        :return:
        """

        # test faking!

        # create a model and store it.
        tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(tmpDir.name)  # used throughout.
        optclim3 = Model.expand('$OPTCLIMTOP/OptClimVn3/')
        refDir = optclim3/'configurations/example_Model/reference'
        post_process = dict(script=optclim3/'scripts/comp_obs.py', output_file='sim_obs.json')
        self.post_process = post_process
        eng = engine.abstractEngine.create_engine('SGE')
        self.engine = eng
        self.model = myModel(name='test_model', reference=refDir,
                             model_dir=testDir, post_process=post_process,
                             parameters=dict(RHCRIT=2, VF1=2.5, CT=2),
                             engine=eng)
        self.tmpDir = tmpDir
        self.testDir = testDir  # for clean up!
        self.refDir = refDir
        self.config_path = self.model.config_path
        self.eng = engine.abstractEngine.create_engine('SGE')

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        shutil.rmtree(self.testDir, onerror=genericLib.errorRemoveReadonly)
        self.tmpDir.cleanup()


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


    def test_inherit(self):
        """
        Test that class inheritance and naming works
        :return:
        """
        # remove any registered classes **except** Mode and myModel. Don't know why I need
        # to do this here as do it above..
        for k in Model.known_models():
            if k not in ["Model",'myModel']:
                Model.remove_class(k)

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
        init1 = dict(
            post_process=dict(high='five', script='$OPTCLIMTOP/OptClimVn3/scripts/comp_obs.py', outputPath='obs.json'),
            parameters=dict(harry=2, fred=oh))
        init2 = copy.copy(init1)
        init2['post_process']['high'] = 'four'
        init2['parameters'] = dict(harry=2, fredSmith=2, fred=oh)
        for name in ['Model', 'model1', 'model2', 'model3', 'model2mod', 'model4']:
            t = Model.model_init(name, name, self.refDir, model_dir=self.testDir / name, **init1)
            t2 = Model.model_init(name, name, self.refDir, model_dir=self.testDir / name, **init2)
            self.assertTrue(type(t) == type(t2), 'Types differ')
            self.assertEqual(t.class_name(), type(t).__name__)
        for name in ['unknown', 'model3aa']:  # unknown models should raise exceptions
            with self.assertRaises(ValueError):
                t = Model.model_init(name, name, self.refDir, model_dir=self.testDir / name, **init1)

    def test_add_param_info(self):

        nl1 = namelist_var(filepath=pathlib.Path('fred'), namelist='james', nl_var='harry')
        nl2 = namelist_var(filepath=pathlib.Path('fred'), namelist='james', nl_var='james')

        expect_param_info = [nl1, nl2]
        pardict = dict(fred=2, james=3)
        model = Model.model_init('myModel', 'test_model',self.refDir, post_process=self.post_process,
                                 model_dir=self.testDir, parameters=pardict,)
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
        model = Model('test_model', self.refDir,post_process=self.post_process,
                      model_dir=self.testDir, parameters=pardict)
        cmd = [model.expand(self.post_process['script']),'input.json',self.post_process['output_file']]
        expected_dct = dict(name='test_model', reference=self.refDir,
                            model_dir=self.testDir, parameters=pardict,
                            post_process={},  _output={},
                            _post_process_input='input.json',
                            _post_process_output='sim_obs.json',
                            post_process_cmd_script=cmd, fake=False, simulated_obs=None,
                            perturb_count=0, parameters_no_key= {},config_path=self.testDir / "test_model.mcfg",
                            status='CREATED', _history=model._history,engine=None,pp_jid=None,run_info={},model_jids=[],
                            submission_count=0,continue_script=pathlib.Path('continue.sh'),
                            submit_script=pathlib.Path('submit.sh'),submitted_jid=None,
                            set_status_script= self.model.expand("$OPTCLIMTOP/OptClimVn3/scripts/set_model_status.py"))

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
            model = Model.model_init(class_name, f'test_{class_name}01', self.refDir,post_process=self.post_process,
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
        nhist = len(self.model._history)
        for status in ['INSTANTIATED', 'SUBMITTED', 'RUNNING', "FAILED", "PERTURBED", "SUBMITTED", "RUNNING",
                       'SUCCEEDED', 'PROCESSED']:
            time.sleep(1e-3)  # sleep for a millisecond
            omodel = copy.deepcopy(self.model)
            with self.assertLogs(level=logging.DEBUG) as log:
                self.model.set_status(status)
            # only check first log entry which is the status change.
            self.assertEqual(log.output[0],
                             f"DEBUG:OPTCLIM.Model:Changing status from {omodel.status} to {self.model.status}")
            # check status is as expected
            self.assertEqual(self.model.status, status)
            nhist += 1  # 1 more history entry
            # verify all but status and history are the same form model prior to status change,
            omodeld = vars(omodel)
            modeld = vars(self.model)
            keys_to_check = set(omodeld.keys()) - {'_history', 'status'}
            for key in keys_to_check:
                self.assertEqual(modeld[key], omodeld[key])
            for key in ['_history', 'status']:  # should be different
                self.assertNotEqual(modeld[key], omodeld[key])
            # history only differs in last entry
            h = copy.deepcopy(modeld['_history'])
            h.popitem()
            self.assertEqual(h, omodeld['_history'])

            # read in the model
            lmodel = Model.load_model(self.config_path)
            self.assertEqual(vars(lmodel), vars(self.model))  # check they are the same
            self.assertEqual(nhist, len(self.model._history))  # history right length
            # test failures

            with self.assertRaises(ValueError):
                self.model.set_status("CREATED")
            with self.assertRaises(ValueError):
                self.model.set_status("FLARTIBARTFAST")


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
        self.assertNotEqual(dd.pop('_history'), dd2.pop('_history'))
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

    # def test_setup_model_env(self):
    #     # create dir and fake config.
    #     self.model.model_dir.mkdir(exist_ok=True,parents=True)
    #     self.model.config_path.touch()
    #     self.model.setup_model_env()
    #     # check environ as expected.
    #     self.assertEqual(os.environ['OPTCLIM_MODEL_PATH'],str(self.model.config_path))
    #     # and file generated contains expected content.
    #     config_pth = self.model.model_dir/'OPTCLIM_MODEL_PATH.json'
    #     with open(config_pth,'rt') as fp:
    #         dct = generic_json.load(fp)
    #     self.assertEqual(dct['config_path'], self.config_path)

    @unittest.mock.patch.object(myModel, 'now', side_effect=gen_time())
    def test_submit_model(self,mck_now):
        """
        Test submit_model works.
         Test status outside expected fails
        :return:
        """
        model = self.model
        model.status = 'INSTANTIATED'  # should have state instantiated
        # need to patch subprocess.check_output
        with unittest.mock.patch('subprocess.check_output', autospec=True,
                                 return_value="Your job 123456") as mock_chk:
            ## testing submission (not continuation)
            result = model.submit_model()
            # expect result to be "123456" # a fake jobid
            self.assertEqual(result, "123456")
            mock_chk.assert_called()  # actually got called
            self.assertEqual(model.pp_jid,'123456')

            # args is a tuple of the arguments (just one list in this case)
            name =  f"{model.name}{len(model.model_jids):05d}"
            outdir = model.model_dir / 'model_output'
            scmd = (self.eng.submit_cmd([str(model.model_dir/'submit.sh')],name,
                                           rundir=model.model_dir,
                                           outdir=outdir,time=2000),)
            self.assertEqual(mock_chk.call_args.args,scmd)


            # also expect changes in status & history
            self.assertEqual(len(model._history), 2)
            k, v = model._history.popitem()  # remove last history entry.
            self.assertEqual(v, [f"Status set to SUBMITTED in {model.model_dir}"])
            # now check that a post-processing script got submitted.
            sub_script = [str(model.set_status_script),str(model.config_path),'PROCESSED']
            expect_output = dict(cmd=self.eng.submit_cmd(sub_script,f"PP_{model.name}",
                                                            outdir = model.model_dir/'PP_output',
                                                            rundir= model.model_dir,
                                                            time=1800,hold=True),result='Your job 123456')
            got = list(model._output.values())[0][0]
            self.assertEqual(got,expect_output)

        # Now test setting status to CONTINUE and test that works.
        # Expect a continue.sh script
        with unittest.mock.patch('subprocess.check_output', autospec=True,
                                 return_value="Your job 123457") as mock_chk:

            model.status = "CONTINUE"
            result = model.submit_model()
            self.assertEqual(model.pp_jid,'123456') # should not change
            self.assertEqual(result,None) # continuing model so now jid!
            mock_chk.assert_called()  # actually got called

            name =  f"{model.name}{len(model.model_jids):05d}"
            outdir = model.model_dir / 'model_output'
            scmd = (self.eng.submit_cmd([str(model.model_dir/'continue.sh')],name,
                                           rundir=model.model_dir,
                                           outdir=outdir,time=2000),)
            self.assertEqual(mock_chk.call_args.args, scmd)


            # also expect changes in status & history,
            self.assertEqual(len(model._history), 2)
            k, v = model._history.popitem()  # remove last history entry.
            self.assertEqual(v, [f"Status set to SUBMITTED in {model.model_dir}"])
            expect_output = dict(cmd=scmd[0],  result='Your job 123457')
            got = list(model._output.values())[-1][0] # TODO fixme This test failing.
            self.assertEqual(got, expect_output)

            # test that submit_cmd works. Will still be continuing!
            mock_chk.reset_mock()
            model.status = "CONTINUE"
            result = model.submit_model()
            mock_chk.assert_called()  # actually got called

            # test that fake-fn does not submit and puts some results in,
            mock_chk.reset_mock()
            model.status = "INSTANTIATED"
            model.pp_jid = None # reset this to None
            result = model.submit_model(fake_function=fake_function)
            # nothing submitted so result and model.pp_jid should be None.
            self.assertIsNone(result)
            self.assertIsNone(model.pp_jid)
            mock_chk.assert_not_called()
            expect = fake_function(model.parameters).rename(model.name)
            pdtest.assert_series_equal(model.simulated_obs, expect)
            self.assertEqual(model.status, 'PROCESSED')

    @unittest.mock.patch.object(myModel, 'now', side_effect=gen_time())
    def test_running(self,mck_now):
        """
        Test running.
        Changes after wards is expecting a file.
        :return:
        """
        # set up vars for grabbing ID
        vars = ['JOB_ID','SLURM_JOB_ID']
        for v in vars:
            os.environ[v]= '123456'
        model = self.model
        model.status = 'SUBMITTED'
        model.running()
        self.assertEqual(model.status, 'RUNNING')
        self.assertEqual(len(model._history), 2)  # should be two entries.
        dmodel = Model.load_model(model.config_path)
        self.assertEqual(dmodel, model)
        # also expect model.model_jids to contain extra ID
        self.assertEqual(model.model_jids,['123456'])

    def test_guess_failed(self):
        """
        Check guess_failed works!
        :return:
        """
        model = copy.deepcopy(self.model)
        model.status="RUNNING"
        model.model_jids=['23456']
        engines =[engine.sge_engine,engine.slurm_engine]

        for eng in engines:
            with unittest.mock.patch.object(eng,'job_status',return_value='RUNNING') as mck:
                model.engine=eng()
                model.guess_failed()
                self.assertEqual(model.status, "RUNNING")

        for eng in engines:
            with unittest.mock.patch.object(eng,'job_status',return_value='notFound') as mck:
                model.status = "RUNNING"
                model.engine=eng()
                model.guess_failed()
                self.assertEqual(model.status, "FAILED")

        for eng in engines:
            with unittest.mock.patch.object(eng,'job_status',return_value='notFound') as mck:
                model.status = "INSTANTIATED"
                model.engine=eng()
                model.guess_failed()
                self.assertEqual(model.status, "INSTANTIATED")



    @unittest.mock.patch.object(myModel, 'now', side_effect=gen_time())
    def test_perturb(self,mck_now):
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
        v = model.read_values('VF1')
        v['VF1'] *= (1 + 1e-7)  # small perturb
        with self.assertLogs(level='DEBUG') as log:
            model.perturb(v)
        self.assertEqual(log.output[-1], f"DEBUG:OPTCLIM.Model: parameters_no_key is now {v}")
        self.assertEqual(len(model._history),    5)
        # expect 5 bits of history. Created, Modified,Instantiated, perturbed using and setting status
        p = model.read_values('VF1')
        self.assertEqual(p, v)
        self.assertEqual(model.perturb_count, 1)  # perturbed it once.
        self.assertEqual(model.status, 'PERTURBED')

    @unittest.mock.patch.object(myModel, 'now', side_effect=gen_time())
    def test_set_failed(self,mck_now):
        """
        Should set status to FAILED.
        :return:
        """


        model = self.model
        model.status = 'RUNNING'
        model.set_failed()
        self.assertEqual(model.status, 'FAILED')
        self.assertEqual(len(model._history), 2)  # should be two entries.

    @unittest.mock.patch.object(myModel,'now',side_effect=gen_time())
    def test_succeeded(self,mock_now):
        """
        Test that succeeded worked! Will try to run a script which we mock.
        :return:
        """
        model = self.model
        model.status = 'RUNNING'  # should be RUNNING
        model.pp_jid='123456'

        with unittest.mock.patch('subprocess.check_output', autospec=True, return_value='Ran PP') as mock_chk:
            with self.assertLogs() as log:
                r = model.succeeded()
            expected = self.eng.release_job(model.pp_jid)
            self.assertEqual(log.output[0], f'INFO:OPTCLIM.Model:Ran post-processing cmd {expected}')
            self.assertEqual(model.status, 'SUCCEEDED')
            self.assertEqual(r, 'Ran PP')
            self.assertEqual(len(model._history), 2)
            dmodel = Model.load_model(model.config_path)
            self.assertEqual(dmodel, model)
            self.assertEqual(len(model._output),1)


            # set pp_jid to None. No subprocess should be submitted
            mock_chk.reset_mock()
            model.pp_jid = None
            model.status = 'RUNNING'
            r = model.succeeded()
            self.assertIsNone(r)


    def test_process(self):
        """
        Test process works.
        Will do first by faking things!
        :return:
        """
        model = Model('test001', self.refDir, post_process=self.post_process,model_dir=self.testDir)
        model.fake = True
        model.status = 'SUCCEEDED'  # we have succeeded
        model.process()  # with fake
        self.assertEqual(model.status, 'PROCESSED')
        # no fake fn..
        model.fake = False
        model.status = 'SUCCEEDED'  # we have succeeded
        # use fake_fn to generate some fake obs!
        fake_obs = self.fake_fn().to_dict()
        # and write them out for
        with open(model.model_dir / model._post_process_output, 'w') as fp:
            generic_json.dump(fake_obs, fp)

        with unittest.mock.patch('subprocess.check_output',
                                 autospec=True, return_value="Submitted something"):
            model.process()  # run the post-processing. Nothing should be ran because of the mock
              # But then nothing can be read in. Need to mock that too? Better to mock the output so
              # it just writes info to file. #TODO modify mock so it writes to model._post_process_output
        pdtest.assert_series_equal(pd.Series(fake_obs).rename(model.name), model.simulated_obs)
        self.assertEqual(model.status, 'PROCESSED')

    # patching end_to_end so time always ticks in controlled way. gen_time does 1 seconds increments.
    @unittest.mock.patch.object(myModel,'now',side_effect=gen_time())
    def test_end_to_end(self,mock_cfg):
        """
        Test that end to end case works.
        :return:
        """

        post_process = dict(script='$OPTCLIMTOP/OptClimVn3/Models/scripts/pp_script_test.py',
                            outputPath='obs.json',
                            fake_obs=self.fake_fn().to_dict())
        cfg = self.testDir / 'model0001.mcfg'
        with unittest.mock.patch('subprocess.check_output', autospec=True,
                                 return_value="Submitted something 56467"):
            with unittest.mock.patch('engine.sge_engine.my_job_id',autospec=True,
                                     return_value="123456"):
                model = myModel('test_model001', self.refDir, model_dir=self.testDir,
                                config_path=cfg,
                                parameters=dict(VF1=1, CT=2.2, G0=11),
                                engine=self.eng,
                                post_process=post_process)  # create the model.

                model.instantiate()  # instantiate the model.
                self.assertIsNotNone(model.post_process_cmd_script)
                model.submit_model()  # submit the model.
                model.running()  # model is running.
                model.succeeded()  # model has succeeded
                # use fake_fn to generate some fake obs!
                fake_obs = self.fake_fn().to_dict()
                # and write them out for
                with open(model.model_dir / model._post_process_output, 'w') as fp:
                    generic_json.dump(fake_obs, fp)
                model.process()  # and do the post-processing
                # need to run a bunch of tests here.
                # having got to here should have simulated_obs be post_process['fake_obs']
                pdtest.assert_series_equal(model.simulated_obs, pd.Series(post_process['fake_obs']).rename(model.name))
                # should have 7 history entries.
                self.assertEqual(len(model._history), 7)
                # four    outputs -- from  model submission, post_process submission, post-process release_job and
                # running post-processing.
                self.assertEqual(len(model._output), 4)

                # now do but where model fails, get perturbed, gets continued and then works.
                cfg = self.testDir / 'model0002.mcfg'
                model = myModel('test_model01', self.refDir, model_dir=self.testDir,
                                config_path=cfg,
                                engine = self.eng,
                                parameters=dict(VF1=1, CT=2.2, G0=11),
                                post_process=post_process)  # create the model.
                model.instantiate()  # instantiate the model.
                model.submit_model()  # submit the model.
                model.running()
                model.set_failed()
                ct = model.read_values('CT')
                print(ct)
                ct['CT'] *= (1 + 1e-6)
                model.perturb(ct)
                model.continue_simulation()
                model.submit_model()
                model.running()
                model.succeeded()  # model has succeeded
                # use fake_fn to generate some fake obs!
                fake_obs = self.fake_fn().to_dict()
                # and write them out for
                with open(model.model_dir / model._post_process_output, 'w') as fp:
                    generic_json.dump(fake_obs, fp)
                model.process()  # and do the post-processing
                # should have 13 history entries.
                self.assertEqual(len(model._history), 13)
                # five  outputs -- 1 model, one continue and one postprocess submission, one post-process release_job and running
                # post-process script.
                self.assertEqual(len(model._output), 5)
            #

    def test_set_post_process(self):
        # tests for set_post_process
        model = Model('fred',self.refDir,post_process=self.post_process)
        pp = copy.deepcopy(self.post_process)
        script = model.expand(pp.pop('script'))
        output = pp.pop('output_file','sim_obs.json')
        input = pp.pop('input_file','input.json')
        # test simple thing.
        self.assertEqual(model.post_process_cmd_script,[script,input,output])
        self.assertEqual(model._post_process_output,output)
        self.assertEqual(model._post_process_input,input)
        self.assertEqual(model.post_process,pp)
        # with interp
        pp=copy.deepcopy(self.post_process)
        pp['interp']='python'
        model.set_post_process(pp)
        self.assertEqual(model.post_process_cmd_script,['python',script,input,output])
        # No PP
        model = Model('fred', self.refDir)
        self.assertEqual(model.post_process_cmd_script,None)
        # set del pp['script']. Should give an error
        pp=copy.deepcopy(self.post_process)
        pp.pop('script')
        with self.assertRaises(ValueError):
            model = Model('fred',self.refDir,post_process=pp)

    def test_key(self):
        """
        Tests for key
        :return:
        """

        # test key for mixed params is as expected
        pDict = {'zz': 1.02, 'aa': 1, 'nn': [0, 1]}
        expect = str(('aa', '1', 'nn', '[0, 1]', 'zz', '1.02'))
        self.model.parameters=pDict
        key = self.model.key()
        self.assertEqual(key, expect)
        # test that small real differences don't cause any differences.
        pDict = {'zz': 1.0200001, 'aa': 1, 'nn': [0, 1]}
        self.model.parameters=pDict
        key = self.model.key()
        self.assertEqual(key, expect)


if __name__ == '__main__':
    unittest.main()
