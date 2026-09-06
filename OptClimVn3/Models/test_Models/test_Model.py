import copy
import datetime
import filecmp
import logging
import os
import pathlib
import shutil
import tempfile
import time
import unittest
import unittest.mock
import tarfile
import shlex

import warnings

import Model
import StudyConfig  # so can read in a config for fake_fn.
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
import xarray
import engine
import genericLib
import generic_json
from namelist_var import NamelistVar
from test_Models.myModel import myModel # needed when testing in linux..

genericLib.setup_env()
def gen_time():
    # used to mock Model.now()
    time = datetime.datetime(2000, 1, 11, 0, 0, 0)
    timedelta = datetime.timedelta(seconds=1)
    while True:
        time += timedelta
        yield time


# To get log info set --log-cli-level WARNING in "additional arguments " in pycharm config
# remove any registered classes **except** myModel.
for k in myModel.known_models():
    if k != "myModel":
        myModel.remove_class(k)



root_pth = myModel.expand("$OPTCLIMTOP/OptClimVn3")
config = StudyConfig.readConfig(root_pth / "configurations/dfols14param_opt3.json")


def fake_function(param):
    return genericLib.fake_fn(config, param)


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
        optclim3 = myModel.expand('$OPTCLIMTOP/OptClimVn3/')
        refDir = optclim3 / 'configurations/example_Model/reference'
        post_process = dict(script=optclim3 / 'scripts/comp_obs.py', output_file='sim_obs.json')
        self.post_process = post_process
        eng = engine.abstractEngine.create_engine('SGE')
        self.engine = eng
        cfg_path = testDir / 'study/test_model.mcfg'
        self.model = myModel(config_path=cfg_path, reference=refDir,
                             post_process=post_process,
                             parameters=dict(RHCRIT=2, VF1=2.5, CT=2,G0=10,ANVIL_FACTOR=0.5,multi_var=2.0),
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


    def test_init(self):
        """
        Test that model init works
        :return:
        """
        pardict = dict(fred=2, james=3)
        config_path = self.testDir / 'study/test_model.mcfg'
        model = Model.Model( config_path=config_path,reference=self.refDir,  post_process=self.post_process,
                             parameters=dict(RHCRIT=2, VF1=2.5, CT=2,G0=10,ANVIL_FACTOR=0.5,multi_var=2.0),
                             engine=self.engine)
        self.assertEqual(model.name, 'test_model')
        self.assertEqual(model.reference_name,self.refDir.name)

        model = Model.Model( config_path=config_path,reference=self.refDir, reference_name='control', post_process=self.post_process,
                             parameters=dict(RHCRIT=2, VF1=2.5, CT=2,G0=10,ANVIL_FACTOR=0.5,multi_var=2.0),
                             engine=self.engine)

        self.assertEqual(model.reference_name, 'control')



    def test_inherit(self):
        """
        Test that class inheritance and naming works
        :return:
        """
        # remove any registered classes **except** Mode and myModel. Don't know why I need
        # to do this here as do it above..
        for k in myModel.known_models():
            if k not in ['myModel']:
                myModel.remove_class(k)

        # define a bunch of sub-classes to check all works

        class model1(myModel):

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
        self.assertEqual(['myModel', 'model1', 'model2', 'model2mod', 'model3', 'model4'],
                         myModel.known_models(), )
        oh = 2
        init1 = dict(
            post_process=dict(high='five', script='$OPTCLIMTOP/OptClimVn3/scripts/comp_obs.py', outputPath='obs.json'),
            parameters=dict(harry=2, fred=oh))
        init2 = copy.copy(init1)
        init2['post_process']['high'] = 'four'
        init2['parameters'] = dict(harry=2, fredSmith=2, fred=oh)
        for name in ['myModel', 'model1', 'model2', 'model3', 'model2mod', 'model4']:
            config_path = self.testDir / f"{name}_1.mcfg"
            t = myModel.model_init(name, config_path=config_path,reference=self.refDir, **init1)
            config_path = self.testDir / f"{name}_2.mcfg"
            t2 = myModel.model_init(name,  reference=self.refDir, config_path=config_path, **init2)
            self.assertTrue(type(t) == type(t2), 'Types differ')
            self.assertEqual(t.class_name(), type(t).__name__)
        for name in ['unknown', 'model3aa']:  # unknown models should raise exceptions
            config_path = self.testDir / f"{name}.mcfg"
            with self.assertRaises(ValueError):
                t = myModel.model_init(name, reference=self.refDir, config_path=config_path, **init1)

    def test_add_param_info(self):

        nl1 = NamelistVar(type_name='json_nl',filepath=pathlib.Path('fred'), namelist='james', nl_var='harry')
        nl2 = NamelistVar(type_name='json_nl',filepath=pathlib.Path('fred'), namelist='james', nl_var='james')

        expect_param_info = [nl1, nl2]
        pardict = dict(fred=2, james=3)
        config_path = self.testDir / 'test_model.mcfg'
        model = myModel.model_init('myModel', config_path=config_path, reference=self.refDir, post_process=self.post_process,
                                  parameters=pardict, )
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
        config_path = self.testDir / 'test_model.mcfg'
        model = myModel( reference=self.refDir, post_process=self.post_process,
                      config_path=config_path, parameters=pardict)
        cmd = [model.expand(self.post_process['script']), 'input.json', self.post_process['output_file']]
        ref=pathlib.PurePath(self.refDir)
        expected_dct = dict(reference=ref,reference_name=ref.name,
                             _config_dir=None,
                            _name="test_model",
                            parameters=pardict,
                            post_process={}, _output={},
                            _post_process_input='input.json',
                            _post_process_output='sim_obs.json',
                            configs=model.configs,
                            post_process_cmd_script=cmd, fake=False, simulated_obs=None,
                            perturb_count=0, parameters_no_key={}, config_path=pathlib.PurePath(self.testDir / "test_model.mcfg"),
                            status='CREATED', _history=model._history, engine=model.engine, pp_jid=None, run_info={},
                            study_properties={},
                            model_jids=[],
                            submission_count=0, continue_script=pathlib.PurePath('continue.sh'),
                            submit_script=pathlib.PurePath('submit.sh'), submitted_jid=None,
                            set_status_script=pathlib.PurePath(self.model.expand("$OPTCLIMTOP/OptClimVn3/scripts/set_model_status.py")),
                            remote=dict(remote_machine=None,remote_model_dir=None),  StudyConfig_path=None,
                            serialisation_data_version=str(model.serialisation_data_version)
                            )

        dct = model.to_dict()

        self.assertEqual(expected_dct, dct)

    def test_load_dump(self):
        """
        Test we can dump a Model and that loading it gives us the same thing.
        Dict contains what we expect!
        :return:
        """
        pardict = dict(fred=2, james=3)

        class model4(myModel):
            def fred(self, message):
                print(f"Fred says: {message}")

        class model5(model4):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fredv = 10

        for class_name in ['myModel', 'model4', 'model5']:
            config_path = self.testDir / f"{class_name}01.mcfg"
            model = myModel.model_init(class_name, reference=self.refDir, post_process=self.post_process,
                                     config_path=config_path, parameters=pardict)
            model.dump_model()
            lmodel = myModel.load_model(model.config_path)
            self.assertEqual(lmodel.class_name(), class_name)
            self.assertEqual(model.to_dict(), lmodel.to_dict())
        lmodel.fred("Hello")
        self.assertEqual(lmodel.fredv, 10)



    def test_set_params(self):
        # test setting params works.
        shutil.copytree(self.refDir, self.model.model_dir, symlinks=True, dirs_exist_ok=True)

        self.model.set_params()
        # use gen_params to get parameters and then check they are as expected.
        nl_iter = self.model.gen_params()
        for (nl, value) in nl_iter.items():
            got = self.model.configs.read_value(nl)
            self.assertEqual(value, got)



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


            self.model.set_status(status)

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
            lmodel = myModel.load_model(self.config_path)
            self.assertEqual(lmodel.to_dict(), self.model.to_dict())  # check they are the same
            self.assertEqual(nhist, len(self.model._history))  # history right length
            # test failures

            with self.assertRaises(ValueError):
                self.model.set_status("CREATED")
            with self.assertRaises(ValueError):
                self.model.set_status("FLARTIBARTFAST")

    def test_read_params(self):
        """
        Test can read param values.
        :return:
        """
        #model = copy.deepcopy(self.model)
        self.model.instantiate() # instantiate the model
        # expect values to be as defined!
        expect_dir = dict(RHCRIT=2, VF1=2.5, CT=2, G0=10)

        got = self.model.read_params(list(expect_dir.keys()))
        self.assertEqual(expect_dir, got)

    @unittest.mock.patch("Model.Model.run_cmd", autospec=True)
    def test_instantiate(self,mck_run_cmd):
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
        self.assertEqual(mck_run_cmd.call_count, 0)  # should not call run_cmd at all -- only called to remote install.

        mm = myModel.load_model(self.config_path)
        self.assertEqual(mm.to_dict(), self.model.to_dict())
        dd = self.model.to_dict()
        dd2 = omodel.to_dict()
        dd2['status'] = 'INSTANTIATED'
        self.assertNotEqual(dd.pop('_history'), dd2.pop('_history'))
        self.assertEqual(dd, dd2)
        count_config = 0
        bak_count = 0
        # how many .bak files do we expect?
        nls = self.model.gen_params()  # gt the namelist files that have changed.
        if self.model.config_dir is None:
            root = self.model.model_dir
        else:
            root = self.model.model_dir/self.model.config_dir
        nl_changed_files = set([root/d.filepath for d in nls.keys()]) # files that are changed and so should have .bak files.

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
                #self.assertTrue(nl_changed_files[file])
            else:
                # see if we have a .bak file.
                bak_file = file.parent / (file.name + ".bak")
                if bak_file.exists():  # got it -- skip
                    continue
                else:
                    self.assertTrue(filecmp.cmp(file, self.model.reference / file.name))

        self.assertEqual(bak_count, expected_bak_count)
        self.assertEqual(1, count_config)  # only one config file.
        ##
        # test. Need to set up remote_machine and remote_dir in model.run_info
        # If remote_dir is not set then just copy to remote model_dir?
        # expect that call run_cmd once and it has the rsync command in it.
        config_path = self.testDir / "study2/test_model.mcfg"
        model = myModel( reference=self.refDir,
                             config_path=config_path, post_process=self.post_process,
                             run_info=dict(remote_machine='user@my.remote.machine.ac.uk', remote_model_dir=pathlib.PurePath('/home/user/remote_model_dir')),
                             parameters=dict(RHCRIT=2, VF1=2.5, CT=2,G0=10,ANVIL_FACTOR=0.5,multi_var=2.0),
                             engine=self.eng)

        # need to recreate model as model creation sets everything up.
        model.instantiate()
        self.assertEqual(mck_run_cmd.call_count, 2)
        cmd_args = mck_run_cmd.call_args_list[0].args[1]  # first arg is self getting the remote create dir
        self.assertIn('ssh', cmd_args[0])
        # rysnc command should be in last call
        cmd_args = mck_run_cmd.call_args_list[-1].args[1]  # first arg is self
        self.assertIn('rsync', cmd_args[0])
        self.assertIn('user@my.remote.machine.ac.uk:/home/user/remote_model_dir/study2', cmd_args[-1])  # dest dir





    @unittest.mock.patch.object(myModel, 'now', side_effect=gen_time())
    def test_submit_model(self, mck_now):
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
            self.assertEqual(model.pp_jid, '123456')

            # args is a tuple of the arguments (just one list in this case)
            name = f"{model.name}{len(model.model_jids):05d}"
            outdir = model.model_dir / 'model_output'
            scmd = self.eng.submit_cmd([str(model.model_dir / 'submit.sh')], name,
                                        rundir=model.model_dir,
                                        outdir=outdir, time=2000)
            scmd = [shlex.quote(s) for s in scmd]
            #scmd = (scmd, )


            self.assertEqual(mock_chk.call_args.args[0], scmd)

            # also expect changes in status & history
            self.assertEqual(len(model._history), 2)
            k, v = model._history.popitem()  # remove last history entry.
            self.assertEqual(v, [f"Status set to SUBMITTED in {model.model_dir}"])
            # now check that a post-processing script got submitted.
            sub_script = [str(model.set_status_script), str(model.config_path), 'PROCESSED']
            scmd = self.eng.submit_cmd(sub_script, f"PP_{model.name}",
                                                         outdir=model.model_dir / 'PP_output',
                                                         rundir=model.model_dir,
                                                         time=1800, hold=True)
            scmd = [shlex.quote(s) for s in scmd]
            expect_output = dict(cmd=scmd, result='Your job 123456')
            got = list(model._output.values())[0][0]
            self.assertEqual(got, expect_output)

        # Now test setting status to CONTINUE and test that works.
        # Expect a continue.sh script
        with unittest.mock.patch('subprocess.check_output', autospec=True,
                                 return_value="Your job 123457") as mock_chk:
            model.status = "CONTINUE"
            result = model.submit_model()
            self.assertEqual(model.pp_jid, '123456')  # should not change
            self.assertEqual(result, None)  # continuing model so now jid!
            mock_chk.assert_called()  # actually got called

            name = f"{model.name}{len(model.model_jids):05d}"
            outdir = model.model_dir / 'model_output'
            scmd = self.eng.submit_cmd([str(model.model_dir / 'continue.sh')], name,
                                        rundir=model.model_dir,
                                        outdir=outdir, time=2000)
            lex_scmd = [shlex.quote(s) for s in scmd]
            self.assertEqual(mock_chk.call_args.args[0], lex_scmd)

            # also expect changes in status & history,
            self.assertEqual(len(model._history), 2)
            k, v = model._history.popitem()  # remove last history entry.
            self.assertEqual(v, [f"Status set to SUBMITTED in {model.model_dir}"])
            expect_output = dict(cmd=lex_scmd, result='Your job 123457')
            got = list(model._output.values())[-1][0]  # get last output
            self.assertEqual(got, expect_output)

            # test that submit_cmd works. Will still be continuing!
            mock_chk.reset_mock()
            model.status = "CONTINUE"
            result = model.submit_model()
            mock_chk.assert_called()  # actually got called

            # test that fake-fn does not submit and puts some results in,
            mock_chk.reset_mock()
            model.status = "INSTANTIATED"
            model.pp_jid = None  # reset this to None
            result = model.submit_model(fake_function=fake_function)
            # nothing submitted so result and model.pp_jid should be None.
            self.assertIsNone(result)
            self.assertIsNone(model.pp_jid)
            mock_chk.assert_not_called()
            expect = fake_function(model.parameters).rename(model.name)
            pdtest.assert_series_equal(model.simulated_obs, expect)
            self.assertEqual(model.status, 'PROCESSED')

    @unittest.mock.patch.object(myModel, 'now', side_effect=gen_time())
    def test_running(self, mck_now):
        """
        Test running.
        Changes after wards is expecting a file.
        :return:
        """
        # set up vars for grabbing ID
        variables = ['JOB_ID', 'SLURM_JOB_ID']
        for v in variables:
            os.environ[v] = '123456'
        model = self.model
        model.status = 'SUBMITTED'
        model.running()
        self.assertEqual(model.status, 'RUNNING')
        self.assertEqual(len(model._history), 2)  # should be two entries.
        dmodel = myModel.load_model(model.config_path)
        self.assertEqual(dmodel.to_dict(), model.to_dict())
        # also expect model.model_jids to contain extra ID
        self.assertEqual(model.model_jids, ['123456'])

    def test_guess_failed(self):
        """
        Check guess_failed works!
        :return:
        """
        model = copy.deepcopy(self.model)
        model.status = "RUNNING"
        model.model_jids = ['23456']
        engines = [engine.sge_engine, engine.slurm_engine]

        for eng in engines:
            with unittest.mock.patch.object(eng, 'job_status', return_value='RUNNING') as mck:
                model.engine = eng()
                model.guess_failed()
                self.assertEqual(model.status, "RUNNING")

        for eng in engines:
            with unittest.mock.patch.object(eng, 'job_status', return_value='notFound') as mck:
                model.status = "RUNNING"
                model.engine = eng()
                model.guess_failed()
                self.assertEqual(model.status, "FAILED")

        for eng in engines:
            with unittest.mock.patch.object(eng, 'job_status', return_value='notFound') as mck:
                model.status = "INSTANTIATED"
                model.engine = eng()
                model.guess_failed()
                self.assertEqual(model.status, "INSTANTIATED")

    @unittest.mock.patch.object(myModel, 'now', side_effect=gen_time())
    def test_perturb(self, mck_now):
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
        v = model.read_params('VF1')
        v['VF1'] *= (1 + 1e-7)  # small perturb
        model.perturb(v)
        self.assertEqual(len(model._history), 6)  # should be 6 entries. # 6 history entries.
        # expect 5 bits of history. Created, Modified,Instantiated, perturbed using and setting status
        p = model.read_params('VF1')
        self.assertEqual(p, v)
        self.assertEqual(model.perturb_count, 1)  # perturbed it once.
        self.assertEqual(model.status, 'PERTURBED')

    @unittest.mock.patch.object(myModel, 'now', side_effect=gen_time())
    def test_set_failed(self, mck_now):
        """
        Should set status to FAILED.
        :return:
        """

        model = self.model
        model.status = 'RUNNING'
        model.set_failed()
        self.assertEqual(model.status, 'FAILED')
        self.assertEqual(len(model._history), 2)  # should be two entries.

    @unittest.mock.patch.object(myModel, 'now', side_effect=gen_time())
    def test_succeeded(self, mock_now):
        """
        Test that succeeded worked! Will try to run a script which we mock.
        :return:
        """
        model = self.model
        model.status = 'RUNNING'  # should be RUNNING
        model.pp_jid = '123456'

        with unittest.mock.patch('subprocess.check_output', autospec=True, return_value='Ran PP') as mock_chk:
            r = model.succeeded()
            expected = self.eng.release_job(model.pp_jid)
            self.assertEqual(model.status, 'SUCCEEDED')
            self.assertEqual(r, 'Ran PP')
            self.assertEqual(len(model._history), 2)
            dmodel = myModel.load_model(model.config_path)
            self.model.compare_objects(dmodel)
            print('input ',type(dmodel),' saved ',type(model))
            self.assertEqual(dmodel.to_dict(), model.to_dict())
            self.assertEqual(len(model._output), 1)

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
        config_path = self.testDir / 'test001.mcfg'
        model = myModel(reference=self.refDir, post_process=self.post_process, config_path=config_path)
        model.fake = True
        model.status = 'SUCCEEDED'  # we have succeeded
        model.process()  # with fake
        self.assertEqual(model.status, 'PROCESSED')
        # no fake fn..
        model.fake = False
        model.status = 'SUCCEEDED'  # we have succeeded
        model.simulated_obs=None # reset obs to None
        # use fake_fn to generate some fake obs!
        fake_obs = self.fake_fn().to_dict()
        # and write them out for process to read!
        with open(model.model_dir / model._post_process_output, 'w') as fp:
            generic_json.dump(fake_obs, fp)

        with unittest.mock.patch('subprocess.check_output',
                                 autospec=True, return_value="Submitted something"):
            model.process()  # run the post-processing. Nothing should actually be ran because of the mock though it shoudl read in the data.
        model_obs = model.compute_simulated_observations()
        pdtest.assert_series_equal(pd.Series(fake_obs).rename(model.name), model_obs)
        self.assertEqual(model.status, 'PROCESSED')

    # patching end_to_end so time always ticks in controlled way. gen_time does 1 seconds increments.
    @unittest.mock.patch.object(myModel, 'now', side_effect=gen_time())
    def test_end_to_end(self, mock_cfg):
        """
        Test that end to end case works.
        :return:
        """

        post_process = dict(script='$OPTCLIMTOP/OptClimVn3/Models/scripts/pp_script_test.py',
                            outputPath='obs.json',
                            fake_obs=self.fake_fn().to_dict())
        cfg = self.testDir / 'test_model0001.mcfg'
        with unittest.mock.patch('subprocess.check_output', autospec=True,
                                 return_value="Submitted something 56467"):
            with unittest.mock.patch('engine.sge_engine.my_job_id', autospec=True,
                                     return_value="123456"):
                model = myModel(reference=self.refDir,
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
                pdtest.assert_series_equal(model.compute_simulated_observations(), pd.Series(post_process['fake_obs']).rename(model.name))
                # should have 8 history entries.
                self.assertEqual(len(model._history), 8)
                # four    outputs -- from  model submission, post_process submission, post-process release_job and
                # running post-processing.
                self.assertEqual(len(model._output), 4)

                # now do but where model fails, get perturbed, gets continued and then works.
                cfg = self.testDir / 'model0002.mcfg'
                model = myModel(reference=self.refDir,
                                config_path=cfg,
                                engine=self.eng,
                                parameters=dict(VF1=1, CT=2.2, G0=11),
                                post_process=post_process)  # create the model.
                model.instantiate()  # instantiate the model.
                model.submit_model()  # submit the model.
                model.running()
                model.set_failed()
                ct = model.read_params('CT')
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
                # should have 14 history entries.
                self.assertEqual(len(model._history), 14)
                # five  outputs -- 1 model, one continue and one postprocess submission, one post-process release_job and running
                # post-process script.
                self.assertEqual(len(model._output), 5)
            #

    def test_set_post_process(self):
        # tests for set_post_process
        model = myModel(config_path=self.testDir / 'fred.mcfg', reference=self.refDir, post_process=self.post_process)
        pp = copy.deepcopy(self.post_process)
        script = model.expand(pp.pop('script'))
        output = pp.pop('output_file', 'sim_obs.json')
        input = pp.pop('input_file', 'input.json')
        # test simple thing.
        self.assertEqual(model.post_process_cmd_script, [script, input, output])
        self.assertEqual(model._post_process_output, output)
        self.assertEqual(model._post_process_input, input)
        self.assertEqual(model.post_process, pp)
        # with interp
        pp = copy.deepcopy(self.post_process)
        pp['interp'] = 'python'
        model.set_post_process(pp)
        self.assertEqual(model.post_process_cmd_script, ['python', script, input, output])
        # No PP
        model = myModel(config_path=self.testDir / 'fred.mcfg', reference=self.refDir)
        self.assertEqual(model.post_process_cmd_script, None)
        # set del pp['script']. Should give an error
        pp = copy.deepcopy(self.post_process)
        pp.pop('script')
        with self.assertRaises(ValueError):
            model = myModel(config_path=self.testDir / 'fred.mcfg', reference=self.refDir, post_process=pp)

        # test per reference_name works
        script2 = model.expand('$OPTCLIMTOP/OptClimVn3/Models/scripts/pp_script_test2.py')
        pp = copy.deepcopy(self.post_process)
        pp['interp'] = 'python'
        pp['post_process_for_reference'] = {'control': dict(script=script2,process_options='some process options')}
        expect_pp  = dict(process_options = 'some process options')
        cofig_path = self.testDir / 'model0001.mcfg'
        model = myModel(reference=self.refDir, post_process=pp,reference_name='control',config_path=cofig_path)
        self.assertEqual(model.post_process_cmd_script, ['python',script2, input, output])
        self.assertEqual(model._post_process_output, output)
        self.assertEqual(model._post_process_input, input)
        self.assertEqual(model.post_process, expect_pp)

    def test_attrs_for_key(self):
        """
        Tests for key
        :return:
        """

        # test key for mixed params is as expected
        p_dict = {'zz': 1.02, 'aa': 1, 'nn': [0, 1]}
        expect = p_dict|dict(reference=self.model.reference)
        self.model.parameters = p_dict
        key = self.model.attrs_for_key()
        self.assertEqual(key, expect)


    def test_archive(self):
        # test archiving works
        #  test config file works

        # create archive file.
        archive_file = self.testDir / 'test_archive.tar'
        pp_file = self.model.model_dir / self.model._post_process_output
        self.model.dump_model()
        with tarfile.open(archive_file, "w", dereference=True) as archive:
            self.model.archive(archive, self.testDir,
                               extra_files=[self.model._post_process_output])  # archive the model

        # now can try and read it.
        expected_names = [p.relative_to(self.model.model_dir) for p in [self.model.config_path]]  # ,]]
        with tarfile.open(archive_file, "r") as archive:
            names = [pathlib.Path(n) for n in archive.getnames()]  # list of names
            self.assertEqual(expected_names, names)

        # now create some obs... and test that works.
        import json
        test_obs = dict(obs1=2.2, obs2=1.0, obs4=True)
        with open(pp_file, 'wt') as fp:
            json.dump(test_obs, fp)
        with tarfile.open(archive_file, "w") as archive:
            self.model.archive(archive, self.testDir)  # archive the model
        expected_names = sorted([p.relative_to(self.model.model_dir) for p in [self.model.config_path, pp_file]])
        with tarfile.open(archive_file, "r") as archive:
            names = sorted([pathlib.Path(n) for n in archive.getnames()])  # list of names
            self.assertEqual(expected_names, names)

        # check extra works.
        #  write a text file
        test_file = self.model.model_dir / 'test.txt'
        with open(test_file, 'wt') as fp:
            print("Line 1", file=fp)
            print("Line 2", file=fp)

        expected_names = sorted([p.relative_to(self.model.model_dir) for p in [self.model.config_path, pp_file, test_file]])
        with tarfile.open(archive_file, "w") as archive:
            self.model.archive(archive, self.testDir, extra_files=['test.txt'])  # archive the model
        with tarfile.open(archive_file, "r") as archive:
            names = sorted([pathlib.Path(n) for n in archive.getnames()])  # list of names
            self.assertEqual(expected_names, names)

    def test_copyConfig(self):
        # test copy works

        # easy test. Copy model somewhere else and check they are the same.
        dest_dir = self.testDir / 'copy_model'
        model = self.model
        model.instantiate()
        model.copyConfig(dest_dir)
        # now load in the copied model
        cmodel = myModel.load_model(dest_dir / f"{self.model.name}.mcfg")
        attrs_not_same = ['config_path','_history']
        for attr in vars(model).keys():
            if attr in attrs_not_same:
                continue
            self.assertEqual(getattr(model, attr), getattr(cmodel, attr), f"Attribute {attr} not the same")


        with self.assertRaises(FileExistsError):
            model.copyConfig(model.model_dir)







    def no_test_reprocess(self):
        # test reprocessing works
        logging.warning("test_reprocessing not implemented")
        #raise NotImplementedError






    def test_gen_params(self):
        """
        Test cases:
        - Test with no params. Should get appropriate values back
        - Test with `parameters` containing valid data.
        - Test to ensure `ValueError` is raised for duplicate namelist.
        """
        model = self.model
        shutil.copytree(self.refDir,model.model_dir)
        expected = {
            NamelistVar(name='CT', type_name='namelist_var', filepath=pathlib.Path('CNTLATM'), namelist='SLBC21',
                        nl_var='ct', default=0.0001): model.parameters['CT'],
            NamelistVar(name='VF1', type_name='namelist_var', filepath=pathlib.Path('CNTLATM'), namelist='SLBC21',
                        nl_var='vf1', default=1): model.parameters['VF1'],
            NamelistVar(type_name='namelist_var', filepath=pathlib.Path('CNTLATM'), nl_var='ANVIL_FACTOR',namelist='RUNCNST', default=0.0,name='ANVIL_FACTOR'):model.parameters['ANVIL_FACTOR'],
            NamelistVar(type_name='namelist_var', filepath=pathlib.Path('CNTLATM'),nl_var='RHCRIT', namelist='RUNCNST'):[model.parameters['RHCRIT']]*19,
            NamelistVar(type_name='namelist_var', filepath=pathlib.Path('CNTLATM'), nl_var='RHCRIT2',default=0.8,
                        namelist='RUNCNST',name='RHCRIT2'): model.parameters['RHCRIT'],
            NamelistVar(type_name='namelist_var', filepath=pathlib.Path('CNTLATM'), nl_var='LATITUDE_BAND', namelist='RUNCNST', default=0.0):model.parameters['multi_var'],
            NamelistVar(type_name='namelist_var', filepath=pathlib.Path('CNTLATM'), nl_var='TOWER_FACTOR', namelist='RUNCNST', default=0.0):model.parameters['multi_var'],

            NamelistVar(name='G0', type_name='namelist_var',filepath=pathlib.Path('CNTLATM'), namelist='SLBC21', nl_var='g0', default=10.0): model.parameters['G0'],

            }
        result = model.gen_params()

        self.assertEqual(expected,result)





        # Specify some parameters
        params = dict(ANVIL_FACTOR=0.5,RHCRIT=0.6) # this gives 3 nameslists
        result = model.gen_params(params)
        self.assertEqual(len(result),3)
        for key,v in params.items():
            nl,v2 = model.param(key,v)[0]
            self.assertEqual(result[nl],v2)

        # test for duplicate namelist.
        # that means hacking param_info to have a duplicate namelist.
        model.param_info.param_constructors['ANVIL_FACTOR2']=model.param_info.param_constructors['ANVIL_FACTOR']
        # Could do this properly using register but that stops duplicate nl.
        with self.assertRaises(ValueError):
            model.gen_params(dict(ANVIL_FACTOR=3.0,RHCRIT=0.6,ANVIL_FACTOR2=3.1))
        # delete the duplicate
        model.param_info.param_constructors.pop('ANVIL_FACTOR2')




    def test_read_param(self):
        """
        Test that read_param works.
                Test cases:
        - Test with a RHCRIT  that is callable and get back
        - Test with VF1 that is a `NamelistVar`.
        - Test with an invalid `parameter` to ensure `KeyError` is raised.
        :return:
        """


        with tempfile.TemporaryDirectory() as tmpdir:
            p = pathlib.Path(tmpdir)
            config_path = p / 'fred.mcfg'
            model = myModel(config_path=config_path,reference= myModel.expand("$OPTCLIMTOP/OptClimVn3/configurations/example_Model/reference")
                            , post_process=self.post_process)  # depends on myModel
            model.instantiate()
            self.assertEqual(model.read_param( 'VF1'), 1) # simple param text
            self.assertEqual(model.read_param( 'RHCRIT'), 0.7) # function test
            with self.assertRaises(KeyError): # invalid param
                model.read_param('INVALID')

    def test_param(self):
        """
        Test cases:
        - Test with a `parameter` that returns a callable.
        - Test with a `parameter` that returns a `NamelistVar`.
        - Test to ensure `KeyError` is raised for invalid `parameter`.
        - Test that bad function returns `ValueError`.
        - test that function returning None accpeted and returns empty list.
        """
        # NamelistVar test
        model = self.model
        model.instantiate()
        expected_nl = model.param_info.param_constructors['VF1'][0]
        expected_val = 2.3
        (nl,val) = model.param('VF1',expected_val)[0]
        self.assertEqual(nl,expected_nl)
        self.assertEqual(val,expected_val)
        # callable
        result = self.model.param('RHCRIT',0.5)[0]
        fn = self.model.param_info.param_constructors['RHCRIT'][0]
        self.assertTrue(callable(fn))
        expected = fn(self.model,0.5)
        self.assertEqual(result,expected)

        # invalid parameter
        with self.assertRaises(KeyError):
            self.model.param('INVALID',0.5)

        # add a dogey fn which returns the wrong kind of thing.

        def bad_fn(self,val):
            return 1
        def none_fn(self,val):
            return None

        self.model.param_info.param_constructors['BAD'] = [bad_fn]
        self.model.param_info.param_constructors['NONE'] = [none_fn]
        self.model._known_parameters_cache = None # force a re-read of the parameters.
        with self.assertRaises(ValueError):
            self.model.param('BAD',0.5)

        self.assertEqual(self.model.param('NONE',0.5),[])

    def test_check(self):
        # test check method works.

        time.sleep(0.001)   # sleep for a millisecond so that get two history entries. (and not on at the same time)
        self.model.check()
        self.assertEqual(len(self.model._history), 2) # create + check
        # set up model.set_status_script to something wrong.
        self.model.set_status_script=pathlib.Path('not_a_script.py')
        with self.assertRaises(ValueError) as cm:
            self.model.check()

    def test_reload(self):
        # test reload works.
        model = self.model
        model.instantiate() # instantiate the model.
        cpy_model = copy.deepcopy(model) # copy the model
        # change the status in memory
        model.status='NO WAY'
        model.reload()
        model == cpy_model
        self.assertEqual(model,cpy_model) # should be the same

    """
    AI Prompt/Spec for test_kill.  Will need to mock run_cmd model_check_status and pp_check_status.
    create a model with pp_jid and model_jids setup. 
    Should do the following tests:
    1) Test that model.kill() works when model is in 'RUNNING' state mocking model_check_status and pp_check_status to return 'Running' and 'Queuing' respectively. Returns [model_jids[-1],pp_jid, ] run_cmd called twice 
    2) Test that model.kill() works when model is in 'SUBMITTED' state mocking model_check_status and pp_check_status to return 'Queuing' and 'Queuing' respectively. Returns [model_jids[-1],pp_jid, ]run_cmd called twice
    3) Test that model.kill() works when model is in 'PROCESSED' state mocking model_check_status and pp_check_status to return 'notFound' and 'notFound' respectively. Returns empty list  run_cmd not called
    4) Test that model.kill() works when model is in 'SUCCEEDED' state mocking model_check_status and pp_check_status to return 'notFound and 'Queuing' respectively. Returns [pp_jid] run_cmd called once
    """

    def test_kill(self):
        model = self.model
        model.pp_jid = 'pp123'
        model.model_jids = ['jid1', 'jid2']
        last_jid = model.model_jids[-1]

        # 1) RUNNING: model_job_status='Running', pp_job_status='Queuing'
        model.status = 'RUNNING'
        with unittest.mock.patch.object(model, 'model_job_status', return_value='Running'), \
                unittest.mock.patch.object(model, 'pp_job_status', return_value='Queuing'), \
                unittest.mock.patch.object(model, 'run_cmd', return_value='Run a cmd') as mock_run_cmd:
            killed = model.kill()
            self.assertEqual(killed, [last_jid, model.pp_jid])
            self.assertEqual(mock_run_cmd.call_count, 2)

        # 2) SUBMITTED: model_job_status='Queuing', pp_job_status='Queuing'
        model.status = 'SUBMITTED'
        with unittest.mock.patch.object(model, 'model_job_status', return_value='Queuing'), \
                unittest.mock.patch.object(model, 'pp_job_status', return_value='Queuing'), \
                unittest.mock.patch.object(model, 'run_cmd', return_value='Run a cmd') as mock_run_cmd:
            killed = model.kill()
            self.assertEqual(killed, [last_jid, model.pp_jid])
            self.assertEqual(mock_run_cmd.call_count, 2)

        # 3) PROCESSED: model_job_status='notFound', pp_job_status='notFound'
        model.status = 'PROCESSED'
        with unittest.mock.patch.object(model, 'model_job_status', return_value='notFound'), \
                unittest.mock.patch.object(model, 'pp_job_status', return_value='notFound'), \
                unittest.mock.patch.object(model, 'run_cmd', return_value='Run a cmd') as mock_run_cmd:
            killed = model.kill()
            self.assertEqual(killed, [])
            self.assertEqual(mock_run_cmd.call_count, 0)

        # 4) SUCCEEDED: model_job_status='notFound', pp_job_status='Queuing'
        model.status = 'SUCCEEDED'
        with unittest.mock.patch.object(model, 'model_job_status', return_value='notFound'), \
                unittest.mock.patch.object(model, 'pp_job_status', return_value='Queuing'), \
                unittest.mock.patch.object(model, 'run_cmd', return_value='Run a cmd') as mock_run_cmd:
            killed = model.kill()
            self.assertEqual(killed, [model.pp_jid])
            self.assertEqual(mock_run_cmd.call_count, 1)

    def test_calendar(self):
        """
        Test that calendar works.
        :return:
        """
        model = self.model
        model.instantiate()
        self.assertEqual(model.calendar(),'standard')


    def test_install_remote_command(self):
        """
        Test that install_remote works.

        Does the following tests:
        1) Case works and gives what we expected!
        2) Test that local_root_dir works as expected
        3) If either remote_machine or remote_dir are None then None is returned and no action taken.
        4) Tests that if wrong types passed a value error is raised.

        :return:
        """
        remote_machine = 'some_random_computer'
        remote_dir = pathlib.PurePath('fred/harry')
        ssh_opts = ["-o", "BatchMode=yes", "-o",
                    "StrictHostKeyChecking=yes"]  # run in batch mode with strict host key checking
        ssh_command = "ssh " + " ".join(shlex.quote(opt) for opt in ssh_opts)
        expected_cmd = ['rsync','-a','-q','-e',ssh_command,str(self.model.model_dir)+'/',f"{remote_machine}:{remote_dir.as_posix()}"]

        cmds = self.model.install_remote_command(remote_machine=remote_machine, remote_model_dir=remote_dir)
        self.assertEqual(cmds[1], expected_cmd)



        # remote_machine is None
        result = self.model.install_remote_command(remote_machine=None, remote_model_dir=remote_dir)
        self.assertIsNone(result)


        # remote_model_dir is None
        result = self.model.install_remote_command(remote_machine=remote_machine, remote_model_dir=None)
        self.assertIsNone(result)

        # failure if remote_dir is not a pure path or None
        with self.assertRaises(ValueError) as err:
            result = self.model.install_remote_command(remote_machine=remote_machine, remote_model_dir=str(remote_dir))
        with self.assertRaises(ValueError) as err:
            result = self.model.install_remote_command(remote_machine=123456, remote_model_dir=remote_dir)

    def test_ssh_command(self):
        """
        Test that ssh_command works.

        Does the following tests:
        1) With only cmd provided simple ssh command is returned
        2) If remote_model_dir
        2) If remote_machine is None then input cmd is returned.
        3) If remote_dir is provided then output directories are changed.

        :return:
        """
        remote_machine = 'some_random_computer'
        remote_dir = pathlib.PurePath('fred/harry')
        local_dir = pathlib.PurePath(self.model.model_dir/'fred')
        cmd = ['ls','-l',local_dir]
        expected_cmd = ['ls','-l',pathlib.PurePath(local_dir).as_posix()]
        ssh_cmd = ['ssh','-q','-o','batchmode=yes','-o','StrictHostKeyChecking=yes',remote_machine]
        expected = ssh_cmd + [" ".join(expected_cmd)]

        result = self.model.ssh_command(cmd, remote_machine=remote_machine)
        self.assertEqual(result, expected)

        # remote_machine is None
        result = self.model.ssh_command(cmd, remote_machine=None)
        self.assertEqual(result, cmd)

        # remote_dir provided
        expected_cmd = ['ls','-l',(remote_dir/'fred').as_posix()]
        expected_cmd = ssh_cmd + [" ".join(expected_cmd)]
        result = self.model.ssh_command(cmd, remote_machine=remote_machine, remote_model_dir=remote_dir)
        self.assertEqual(result, expected_cmd)

    def test_update_params(self):
        """"
        Test that update_params works.
        """
        model = self.model
        # try and update but not instantiated should fail
        with self.assertRaises(ValueError):
            got_params = model.update_params(['ENTCOEF', 'ICE_SIZE'])

        # test update params works
        model.instantiate()
        got_params = model.update_params(['ENTCOEF','ICE_SIZE'])
        expect_params = model.parameters.copy()
        expect_params.update(model.read_params(['ENTCOEF','ICE_SIZE']))
        expect_params = pd.Series(expect_params).rename(model.name)


        self.assertTrue(expect_params.equals(got_params.reindex(expect_params.index)))

    def test_update_reference_name(self):
        """
        test that update_reference_name works. Should update the reference name and then update the parameters to match the new reference.
        :return:

        Three tests:
        1) Create a model with reference name 'control' and then update to 'experiment'. Check that warning issued, reference name is updated, Model config file is updated, and there are two more history entries
        2) Update the reference name to the same name. Check that nothing changes and no history entries are added.
        3) update the reference name to 'control2' with dump set to False. Check that reference name is updated, Model config file is not updated, but one history entry is added.
        """
        import tempfile
        import pathlib
        from unittest.mock import patch

        # 1) Create model with reference_name 'control'
        tmpdir = tempfile.TemporaryDirectory()
        model_dir = pathlib.Path(tmpdir.name)
        ref_dir = model_dir / 'ref'
        ref_dir.mkdir()
        config_path = model_dir / 'model_config.json'
        # Create a dummy config file
        config_path.write_text('{}')

        # Create model
        config_path = model_dir / 'model_config.json'
        model = Model.Model(reference=ref_dir, reference_name='control',  config_path=config_path)
        old_history_len = len(getattr(model, 'history', []))
        old_ref = model.reference_name

        # Patch dump to check it is called
        with unittest.mock.patch('Model.Model.dump', autospec=True) as mock_dump:
            with self.assertLogs('OPTCLIM.Model', level='WARNING') as log:
                model.update_reference_name('experiment', dump=True)
            # Check warning issued
            self.assertTrue(any('Updating reference from control to experiment' in msg for msg in log.output))
            # Check reference_name updated
            self.assertEqual(model.reference_name, 'experiment')
            # Check config file updated (dump called)
            mock_dump.assert_called_with(model,config_path)
            # Check two more history entries
            self.assertEqual(len(model._history), old_history_len + 2)

            # 2) Update reference name to same name
            mock_dump.reset_mock()
            old_history_len = len(model._history)
            with self.assertNoLogs('OPTCLIM.Model', level='WARNING') as log:
                model.update_reference_name('experiment', dump=True)
            # No history entries added
            self.assertEqual(len(model._history), old_history_len)
            mock_dump.assert_not_called()

            #3 change reference name but dump False
            time.sleep(1e-3) # history keys are timestamps. This, in retrospect, is a bad idea. Should have a list of times/messages.
            mock_dump.reset_mock()
            old_history_len = len(model._history)
            with self.assertLogs('OPTCLIM.Model', level='WARNING') as log:
                model.update_reference_name('control2', dump=False)
            # Check warning issued
            self.assertTrue(any('Updating reference from experiment to control2' in msg for msg in log.output))
            # Check reference_name updated
            self.assertEqual(model.reference_name, 'control2')
            # Check config file not updated (dump not called)
            mock_dump.assert_not_called()
            # Check one history entry added
            self.assertEqual(len(model._history), old_history_len + 1)

        tmpdir.cleanup()


    def test_compute_simulated_observations(self):
        """
        AI-generated contract-based tests for compute_simulated_observations.

        This method has a narrow set of public contracts, so the suite is intentionally
        small and explicit rather than broad. The test method currently contains nine
        subtests because a few of the original categories are naturally split into separate
        checks for the file layer, cache behaviour, and format-specific parsing:

        1) invalid status short-circuits without reading any file;
        2) JSON success path returns the expected pandas Series;
        3) null-valued observations raise ValueError;
        4) an existing cache is reused when use_cache=True;
        5) stale cache on a non-PROCESSED status raises ValueError;
        6) missing output file raises FileNotFoundError;
        7) unsupported output types raise NotImplementedError;
        8) netCDF parsing returns the expected Series; and
        9) CSV parsing returns the expected Series.

        This keeps the tests anchored to externally visible behaviour without trying to
        exhaustively mirror the implementation details of every branch.

        :return:
        """
        model = self.model

        def run_without_reading(model_to_check, use_cache=True):
            """Helper for assertions that the method must short-circuit before any file I/O."""
            with unittest.mock.patch.object(
                    pathlib.Path, 'is_file',
                    side_effect=AssertionError('is_file should not be called')), \
                 unittest.mock.patch(
                    'builtins.open',
                    side_effect=AssertionError('open should not be called')), \
                 unittest.mock.patch(
                    'json.load',
                    side_effect=AssertionError('json.load should not be called')):
                return model_to_check.compute_simulated_observations(use_cache=use_cache)

        def set_case(status, output_file, simulated_obs=None):
            model.status = status
            model._post_process_output = output_file
            model.simulated_obs = simulated_obs

        # Core contract: invalid state must short-circuit before any file or JSON parsing occurs.
        with self.subTest("invalid status returns None without reading"):
            set_case('CREATED', 'sim_obs.json')
            result = run_without_reading(model)
            self.assertIsNone(result)

        # JSON read path: valid post-processed payloads should become a named Series with the expected values.
        with self.subTest("json success path"):
            set_case('PROCESSED', 'sim_obs.json')
            expected = pd.Series({'alpha': 1.0, 'beta': 2.0}, name=model.name)
            with unittest.mock.patch.object(pathlib.Path, 'is_file', return_value=True), \
                 unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{}')) as open_mock, \
                 unittest.mock.patch('json.load', return_value={'alpha': 1.0, 'beta': 2.0}) as json_load:
                result = model.compute_simulated_observations(use_cache=False)
            pdtest.assert_series_equal(result, expected)
            pdtest.assert_series_equal(model.simulated_obs, expected)
            open_mock.assert_called_once()
            json_load.assert_called_once()

        # Defensive validation: null-valued observations are invalid and must raise a hard failure.
        with self.subTest("null rejection"):
            set_case('PROCESSED', 'sim_obs.json')
            with unittest.mock.patch.object(pathlib.Path, 'is_file', return_value=True), \
                 unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{}')), \
                 unittest.mock.patch('json.load', return_value={'alpha': 1.0, 'beta': None}):
                with self.assertRaisesRegex(ValueError, 'Obs contains null values'):
                    model.compute_simulated_observations(use_cache=False)
            self.assertIsNone(model.simulated_obs)

        # Cache semantics: a valid cached Series should bypass the file-read path completely.
        with self.subTest("cache is used when available"):
            cached = pd.Series({'alpha': 3.0}, name=model.name)
            set_case('PROCESSED', 'sim_obs.json', cached)
            result = run_without_reading(model)
            pdtest.assert_series_equal(result, cached)

        # Guardrail: stale cached values must not be accepted when the model is not PROCESSED.
        with self.subTest("stale cache raises for non-processed status"):
            set_case('SUCCEEDED', 'sim_obs.json', pd.Series({'alpha': 3.0}, name=model.name))
            with self.assertRaisesRegex(ValueError, 'Should not have simulated observations when status is SUCCEEDED'):
                model.compute_simulated_observations(use_cache=True)

        # Error handling: a missing post-processed output is not recoverable and should raise immediately.
        with self.subTest("missing output file raises"):
            set_case('PROCESSED', 'missing.json')
            with unittest.mock.patch.object(pathlib.Path, 'is_file', return_value=False):
                with self.assertRaisesRegex(FileNotFoundError, 'Could not find post-processed file'):
                    model.compute_simulated_observations(use_cache=False)

        # Unsupported-type handling: an unrecognised suffix should fail explicitly rather than returning garbage.
        with self.subTest("unsupported extension raises"):
            set_case('PROCESSED', 'sim_obs.txt')
            with unittest.mock.patch.object(pathlib.Path, 'is_file', return_value=True):
                with self.assertRaisesRegex(NotImplementedError, 'Do not recognize'):
                    model.compute_simulated_observations(use_cache=False)

        # Format support: keep one representative netCDF case to lock down the supported scalar-dataset behaviour.
        with self.subTest("netcdf success path"):
            set_case('PROCESSED', 'sim_obs.nc')
            expected = pd.Series({'alpha': 1.0, 'beta': 2.0}, name=model.name)
            ds = xarray.Dataset({
                'alpha': xarray.DataArray(np.array(1.0)),
                'beta': xarray.DataArray(np.array(2.0)),
            })
            with unittest.mock.patch.object(pathlib.Path, 'is_file', return_value=True), \
                 unittest.mock.patch('xarray.load_dataset', return_value=ds) as load_dataset:
                result = model.compute_simulated_observations(use_cache=False)
            pdtest.assert_series_equal(result, expected)
            load_dataset.assert_called_once()

        # Format support: keep one representative CSV case to lock down the expected pandas conversion semantics.
        with self.subTest("csv success path"):
            set_case('PROCESSED', 'sim_obs.csv')
            expected = pd.Series({'alpha': 1.0, 'beta': 2.0}, name=model.name)
            mock_df = unittest.mock.Mock()
            mock_df.to_dict.return_value = {'alpha': 1.0, 'beta': 2.0}
            with unittest.mock.patch.object(pathlib.Path, 'is_file', return_value=True), \
                 unittest.mock.patch('pandas.read_csv', return_value=mock_df) as read_csv:
                result = model.compute_simulated_observations(use_cache=False)
            pdtest.assert_series_equal(result, expected)
            read_csv.assert_called_once()




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, force=True)
    unittest.main()
