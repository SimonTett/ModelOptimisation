
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

import StudyConfig  # so can read in a config for fake_fn.
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest
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
        self.model = myModel(name='test_model', reference=refDir,
                             model_dir=testDir / 'study', post_process=post_process,
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
            t = myModel.model_init(name, name, self.refDir, model_dir=self.testDir / name, **init1)
            t2 = myModel.model_init(name, name, self.refDir, model_dir=self.testDir / name, **init2)
            self.assertTrue(type(t) == type(t2), 'Types differ')
            self.assertEqual(t.class_name(), type(t).__name__)
        for name in ['unknown', 'model3aa']:  # unknown models should raise exceptions
            with self.assertRaises(ValueError):
                t = myModel.model_init(name, name, self.refDir, model_dir=self.testDir / name, **init1)

    def test_add_param_info(self):

        nl1 = NamelistVar(type_name='json_nl',filepath=pathlib.Path('fred'), namelist='james', nl_var='harry')
        nl2 = NamelistVar(type_name='json_nl',filepath=pathlib.Path('fred'), namelist='james', nl_var='james')

        expect_param_info = [nl1, nl2]
        pardict = dict(fred=2, james=3)
        model = myModel.model_init('myModel', 'test_model', self.refDir, post_process=self.post_process,
                                 model_dir=self.testDir, parameters=pardict, )
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
        model = myModel('test_model', self.refDir, post_process=self.post_process,
                      model_dir=self.testDir, parameters=pardict)
        cmd = [model.expand(self.post_process['script']), 'input.json', self.post_process['output_file']]
        expected_dct = dict(name='test_model', reference=pathlib.PurePath(self.refDir),
                            model_dir=pathlib.PurePath(self.testDir), config_dir=pathlib.PurePath(self.testDir),
                            parameters=pardict,
                            post_process={}, _output={},
                            _post_process_input='input.json',
                            _post_process_output='sim_obs.json',
                            configs=dict(configs={},root_dir=model.model_dir),
                            post_process_cmd_script=cmd, fake=False, simulated_obs=None,
                            perturb_count=0, parameters_no_key={}, config_path=pathlib.PurePath(self.testDir / "test_model.mcfg"),
                            status='CREATED', _history=model._history, engine=model.engine, pp_jid=None, run_info={},
                            model_jids=[],
                            submission_count=0, continue_script=pathlib.PurePath('continue.sh'),
                            submit_script=pathlib.PurePath('submit.sh'), submitted_jid=None,
                            set_status_script=pathlib.PurePath(self.model.expand("$OPTCLIMTOP/OptClimVn3/scripts/set_model_status.py")))

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
            model = myModel.model_init(class_name, f'test_{class_name}01', self.refDir, post_process=self.post_process,
                                     model_dir=self.testDir, parameters=pardict)
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

    @unittest.mock.patch("Model.Model.install_remote", autospec=True)
    def test_instantiate(self,mck_install):
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
        self.assertEqual(mck_install.call_count, 1)  # called install_remote.

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
        nl_changed_files = set([self.model.config_dir/d.filepath for d in nls.keys()]) # files that are changed and so should have .bak files.

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
            #scmd = [shlex.quote(s) for s in scmd]
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
            expect_output = dict(cmd=scmd, result='Your job 123457')
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
        model = myModel('test001', self.refDir, post_process=self.post_process, model_dir=self.testDir)
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
            model.process()  # run the post-processing. Nothing should be ran because of the mock
        pdtest.assert_series_equal(pd.Series(fake_obs).rename(model.name), model.simulated_obs)
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
        cfg = self.testDir / 'model0001.mcfg'
        with unittest.mock.patch('subprocess.check_output', autospec=True,
                                 return_value="Submitted something 56467"):
            with unittest.mock.patch('engine.sge_engine.my_job_id', autospec=True,
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
                # should have 8 history entries.
                self.assertEqual(len(model._history), 8)
                # four    outputs -- from  model submission, post_process submission, post-process release_job and
                # running post-processing.
                self.assertEqual(len(model._output), 4)

                # now do but where model fails, get perturbed, gets continued and then works.
                cfg = self.testDir / 'model0002.mcfg'
                model = myModel('test_model01', self.refDir, model_dir=self.testDir,
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
        model = myModel('fred', self.refDir, post_process=self.post_process)
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
        model = myModel('fred', self.refDir)
        self.assertEqual(model.post_process_cmd_script, None)
        # set del pp['script']. Should give an error
        pp = copy.deepcopy(self.post_process)
        pp.pop('script')
        with self.assertRaises(ValueError):
            model = myModel('fred', self.refDir, post_process=pp)

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
        expected_names = [p.relative_to(self.testDir) for p in [self.model.config_path]]  # ,]]
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
        expected_names = [p.relative_to(self.testDir) for p in [self.model.config_path, pp_file]]
        with tarfile.open(archive_file, "r") as archive:
            names = [pathlib.Path(n) for n in archive.getnames()]  # list of names
            self.assertEqual(expected_names, names)

        # check extra works.
        #  write a text file
        test_file = self.model.model_dir / 'test.txt'
        with open(test_file, 'wt') as fp:
            print("Line 1", file=fp)
            print("Line 2", file=fp)

        expected_names = [p.relative_to(self.testDir) for p in [self.model.config_path, pp_file, test_file]]
        with tarfile.open(archive_file, "w") as archive:
            self.model.archive(archive, self.testDir, extra_files=['test.txt'])  # archive the model
        with tarfile.open(archive_file, "r") as archive:
            names = [pathlib.Path(n) for n in archive.getnames()]  # list of names
            self.assertEqual(expected_names, names)

    def test_copy(self):
        # test copy works
        logging.warning("test_copy not implemented")
        #raise NotImplementedError

    def test_reprocess(self):
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
            model = myModel('fred', myModel.expand("$OPTCLIMTOP/OptClimVn3/configurations/example_Model/reference")
                            , self.post_process, model_dir=p)  # depends on myModel
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

    @unittest.mock.patch("model_base.journal.run_cmd", autospec=True)
    def test_install_remote(self,mck_run_cmd):
        """
        Test that install_remote works.

        Does the following tests:
        1) Case works and gives what we expected!
        2) If either remote_machine or remote_dir are None that mck.call_count is still 1.

        :return:
        """
        remote_machine = 'some_random_computer'
        remote_dir = pathlib.PurePath('fred/harry')
        expected_cmd = ['rsync','-a','-q',f"{self.model.model_dir}",f"{remote_machine}:{remote_dir.as_posix()}/"]

        result = self.model.install_remote(remote_machine=remote_machine, remote_dir=remote_dir)
        self.assertTrue(result)
        self.assertEqual(mck_run_cmd.call_count, 1)
        self.assertEqual(mck_run_cmd.call_args[0][1], expected_cmd)

        # remote_match is None
        result = self.model.install_remote(remote_machine=None, remote_dir=remote_dir)
        self.assertTrue(result)
        self.assertEqual(mck_run_cmd.call_count, 1)

        # remote_dir is None
        result = self.model.install_remote(remote_machine=None, remote_dir=remote_dir)
        self.assertTrue(result)
        self.assertEqual(mck_run_cmd.call_count, 1)

        # failure if remote_dir is not a pure path or None
        with self.assertRaises(ValueError) as err:
            result = self.model.install_remote(remote_machine=remote_machine, remote_dir=str(remote_dir))
        with self.assertRaises(ValueError) as err:
            result = self.model.install_remote(remote_machine=123456, remote_dir=remote_dir)


        



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, force=True)
    unittest.main()
