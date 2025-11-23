"""
Test cases for SubmitStudy classes
"""
import datetime

import pathlib

import tempfile
import unittest.mock  # need to mock the run case.
import unittest

import Study
import StudyConfig
import SubmitStudy
import engine
from Model import Model
import copy
import pandas as pd
import genericLib
genericLib.setup_env()

def gen_time():
    # used to mock Model.now()
    time = datetime.datetime(2000, 1, 11, 0, 0, 0)
    timedelta = datetime.timedelta(seconds=1)
    while True:
        time += timedelta
        yield time


class myModel(Model):
    pass


# class that inherits from Model.
times = gen_time()
pth = myModel.expand("$OPTCLIMTOP/OptClimVn3/Models/parameter_config/example_Parameters.csv")
myModel.update_from_file(pth)


class MyTestCase(unittest.TestCase):

    @unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'now', side_effect=times)  # regen times every time!
    @unittest.mock.patch.object(myModel, 'now', side_effect=times)
    def setUp(self, mck_now, mck_model):
        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(self.tmpDir.name)
        optclim3 = Model.expand('$OPTCLIMTOP/OptClimVn3/')
        refDir = optclim3/'configurations/example_Model'
        cpth = refDir/"configurations/dfols14param_opt3.json"
        refDir = refDir/'reference'
        config = StudyConfig.readConfig(cpth)
        config.baseRunID('ZZ')

        submit = SubmitStudy.SubmitStudy(config, model_name='myModel', rootDir=testDir,next_iter_cmd=['run myself'])
        # create some models
        models=[]
        for param in [dict(VF1=3, CT=1e-4), dict(VF1=2.4, CT=1e-4), dict(VF1=2.6, CT=1e-4)]:
            models.append(submit.create_model(param, dump=True))
        submit.update_iter(models)
        submit.dump_config(dump_models=True)
        self.submit = submit
        self.testDir = testDir

    def tearDown(self):
        self.tmpDir.cleanup()

    def test_create_model(self):
        """
        Test that can create a model.
        :return:
        """
        params = dict(VF1=2.2, ENTCOEF=3)
        paramD = copy.deepcopy(params)
        paramD.update(self.submit.config.fixedParams())
        model = self.submit.create_model(paramD, dump=False)
        self.assertTrue(isinstance(model, Model))
        self.assertEqual(model.parameters, paramD)
        # now create one that goes to disk,
        paramD.update(VF1=2.1)
        model2 = self.submit.create_model(paramD)
        self.assertEqual(model2.parameters,paramD)
        self.submit.dump_config(dump_models=True)
        # as dump_models is True expect SubmitStudy obj on disk and model as well.
        # load them and compare.
        m2 = Model.load_model(model.config_path)
        m3 = Model.load_model(model2.config_path)
        self.assertEqual(model, m2)
        self.assertEqual(model2, m3)
        sub2 = self.submit.load(self.submit.config_path)
        self.assertEqual(self.submit, sub2)

    def test_copy(self):
        # test that we can copy a SubmitStudy object
        submit = self.submit
        submit.instantiate()
        copy_dir = self.testDir / 'copy_test'
        sub2 = submit.copy(copy_dir)
        self.assertIsInstance(sub2,SubmitStudy.SubmitStudy)
        attrs_different = ['rootDir','config_path','_history']
        for attr in vars(submit).keys():
            val1 = getattr(submit,attr)
            val2 = getattr(sub2,attr)
            if attr in attrs_different:
                self.assertNotEqual(val1,val2,msg=f"Attribute {attr} should be different")
            elif attr == 'model_index':
                # keys same, models different because different paths and history changes
                self.assertEqual(set(val1.keys()),set(val2.keys()),msg="Model index keys should be the same")
            else:
                self.assertEqual(val1,val2,msg=f"Attribute {attr} should be the same")


        # make another copy where we update parameters
        params_to_update = ['ENTCOEF','ICE_SIZE']
        copy_dir = self.testDir / 'copy_test2'
        sub3 = submit.copy(copy_dir, update_parameters=params_to_update)

        # check that the parameters have been updated in all models
        param_values = None
        for key,model in sub3.model_index.items():
            self.assertEqual(key, sub3.key_for_model(model))
            if param_values is None:
                param_values = {param: model.parameters[param] for param in params_to_update}
            else:
                got = {param: model.parameters[param] for param in params_to_update}
                self.assertEqual(got,param_values,msg="Parameters should be the same across models")

        # and another copy where paths are not updated! Shoudl eb identical but check have model dirs in copy
        copy_dir = self.testDir / 'copy_test3'
        sub4 = submit.copy(copy_dir, update_paths=False)
        self.assertEqual(submit, sub4)
        for model in sub4.model_index.values():
            model.reload() # force reload which will flush various cached paths

            # and the copy model should be identical.
            pth = copy_dir/(model.config_path.relative_to(submit.rootDir))
            mcopy = Model.load(pth) # using load so no path changes.
            for key in vars(model).keys():
                val1 = getattr(model,key)
                val2 = getattr(mcopy,key)
                if val1 != val2:
                    pass # put breakpoint here
                self.assertEqual(val1,val2,msg=f"Attribute {key} should be identical between model and loaded copy")
            #self.assertEqual(model,mcopy,msg="Model in copy should be identical to loaded model from path")













    @unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'now', side_effect=times)
    def test_delete(self, mck_now):
        # can we delete things.
        pth = self.submit.config_path
        self.submit.config.baseRunID(value='ZZ')
        mpths = [m.config_path for m in self.submit.model_index.values()]
        if not pth.exists():
            raise ValueError(f"config {pth} does not exist")
        for mpth in mpths:
            if not mpth.exists():
                raise ValueError(f"Model config {mpth} does not exist")
        nhist = len(self.submit._history)
        self.submit.delete()  # should delete everything including models.
        # models should all be gone, no model index and next_name be ZZ000
        self.assertFalse(pth.exists())
        for mpth in mpths:
            self.assertFalse(mpth.exists())
        self.assertEqual(self.submit.gen_name(), 'ZZ000')
        self.assertEqual(len(self.submit._history), nhist + 1)  # added deleted

    def test_dump_load(self):
        submit = self.submit

        submit.dump(submit.config_path)
        nsub = SubmitStudy.SubmitStudy.load(submit.config_path)
        self.assertEqual(submit, nsub)  # should be identical
        self.assertEqual(nsub.config._filename, submit.config._filename)

        for m1, m2 in zip(submit.model_index.values(), nsub.model_index.values()):
            self.assertEqual(m1, m2)
        # check on disk is paths
        import json
        with submit.config_path.open('r') as fp:
            dct = json.load(fp)
        for (k1, m1), (k2, m2) in zip(dct['object']['model_index'].items(), submit.model_index.items()):
            self.assertEqual(m1['object'], str(m2.config_path))
            self.assertEqual(k1, k2)

    def test_load_config(self):
        # test some functionality in load_config works.
        pth = self.submit.config_path
        self.submit.dump_config()
        newSub = self.submit.load_SubmitStudy(pth)
        self.assertEqual(self.submit, newSub)
        # explicitly check

        study = self.submit.load_SubmitStudy(pth, Study=True)
        # return as a study
        self.assertIsInstance(study, Study.Study)

    def test_gen_name(self):
        # have generated three models so gen_model should be
        self.submit.name_values = None
        name = self.submit.gen_name()
        self.assertEqual(name, 'ZZ000')
        # set name_values[0] to 11 (should get a)
        self.submit.name_values[0] = 9
        name = self.submit.gen_name()
        self.assertEqual(name, 'ZZ00a')
        # and increase it to 36
        self.submit.name_values = [35, 0, 0]
        name = self.submit.gen_name()
        self.assertEqual(name, 'ZZ010')

        self.submit.name_values = [34, 35, 35]
        name = self.submit.gen_name()
        self.assertEqual(name, 'ZZzzz')

    def test_instantiate(self):
        # test we can instantiate all relevant models.
        # will instantiate  one model directly and then instantiate everything.
        # should have two new models then!
        submit = self.submit
        lst_model = list(submit.model_index.values())[-1]
        lst_model.instantiate()
        # at this point expect model_dir to contain several files. Depends on the model how many. We will just expect
        # more than 3
        self.assertTrue(len(list(lst_model.model_dir.glob("*"))) > 3)

        # and rootdir should contain ONLY model_dirs & config
        pths_got = set(submit.rootDir.glob("*"))
        pths_expect  = [submit.config_path]
        pths_expect += [m.model_dir for m in submit.model_index.values()]
        self.assertEqual(set(pths_got), set(pths_expect))

        # now instantiate. All model dirs should have  > 3 files.
        submit.instantiate()
        for model in submit.model_index.values():
            nfiles = len(list(model.model_dir.glob("*")))
            self.assertTrue(nfiles > 3)
        # and rootdir should contain ONLY model dirs and config.
        pths_got = set(submit.rootDir.glob("*"))
        self.assertEqual(set(pths_got), set(pths_expect))

    # need to mock both SubmitStudy and myModel now.
    @unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'now', side_effect=times)
    @unittest.mock.patch.object(myModel, 'now', side_effect=times)
    def test_submit_all_models(self, mck_now, mck_model_now):

        # set up the fake rtn output
        submit = copy.deepcopy(self.submit)
        job_nos = range(34567,34567+(1+2*len(submit.model_index))) # job numbers
        output = [f"Job submitted {item}" for  item in job_nos]
        # list of sequential jobs.,
        with unittest.mock.patch("subprocess.check_output",
                                 autospec=True, side_effect=output) as mck_output:
            submit.submit_all_models()
            # run submit -- nothing should happen as no models are instantiated
            mck_output.assert_not_called()
            submit.instantiate()  # instantiate all models.
            submit.submit_all_models()
            # run submit -- should submit the three * (pp process and  models). so 6 times
            self.assertEqual(mck_output.call_count, 7)
            # and now run the models -- which means we need to model my_job_id
            models = submit.model_index.values()
            model_job_nos = range(123456, 123456 + len(models))  # job numbers
            with unittest.mock.patch("engine.sge_engine.my_job_id",autospec=True,
                                     side_effect=[str(jid) for jid in model_job_nos]):
                for m in models:
                    m.running()

            # expect that the ob id is every 2nd job_no (as a str)
            # and model_jid is as model_job_nos
            # so lets check that.
            for model,jno,mjid in zip(submit.model_index.values(),job_nos[0::2],model_job_nos):
                self.assertEqual(model.pp_jid,str(jno))
                self.assertEqual(model.model_jids,[str(mjid)])

        # now have the models all fail and then continue.  Should only have three cases -- the models
        # All status should be submitted and the jid cmd should be unchanged
        output_continue = [f"Job submitted {item+100}" for item in job_nos[1::2]]
        with unittest.mock.patch("subprocess.check_output",
                                 autospec=True, side_effect=output_continue) as mck_output:
            for model in submit.model_index.values():
                model.status = "CONTINUE"
            submit.submit_all_models()
            self.assertEqual(mck_output.call_count, 3)  # three cases
            # check models are as expected.
            model_job_nos2 = range(123456+100, 123456+100 + len(models))  # job numbers
            with unittest.mock.patch("engine.sge_engine.my_job_id",autospec=True,side_effect=[str(jid) for jid in model_job_nos2]):
                for indx,model in enumerate(submit.model_index.values()):
                    self.assertEqual(model.status, "SUBMITTED")
                    self.assertEqual(model.pp_jid, str(job_nos[indx*2]))
                    model.running() # model gets to run.
                    mj = [str(model_job_nos[indx]),str(model_job_nos2[indx])] # jid for start job. Continue job should have this +100
                    self.assertEqual(model.model_jids,mj)

        # final actual run test. Turn of next iteration.
        submit.next_iter_cmd=None
        with unittest.mock.patch("subprocess.check_output",
                                 autospec=True, side_effect=output) as mck_output:
            for m in submit.model_index.values():
                m.status = 'INSTANTIATED'
                m.pp_jid = None # set post_process_cmd back to None.
                m.model_jids = [] # and model list to empty
            submit.submit_all_models()
            self.assertEqual(mck_output.call_count, 6)  # 7 cases. 3 x (model + pp submit) + one next job.

        # final tests -- history and output as expected.
        # expect 9 history:
        #    3 x model created,  instantiated,  models submitted,
        #      models continued, models submitted, next job submitted
        #
        # and 3+5 outputs
        self.assertEqual(len(submit._history), 8)
        self.assertEqual(len(submit._output), 1)  # Next iter.

        # now fake it. subprocess.check_output should not run anything.

        def fake_function(param):
            sim_obs = dict()
            obs_count = 0
            for k, v in param.items():
                oname = f"obs{obs_count}"
                try:
                    sim_obs[oname] = v ** 2
                    obs_count += 1
                except TypeError: # multiplication not defined
                    pass
            return pd.Series(sim_obs)

        with unittest.mock.patch("subprocess.check_output",
                                 autospec=True, return_value="some value 345678") as mck_output:
            submit = copy.deepcopy(self.submit)
            submit.instantiate()  # instantiate all models.
            submit.submit_all_models(fake_fn=fake_function)
            mck_output.assert_not_called()

    dt = datetime.datetime(2022, 1, 1, 0, 0, 0)

    @unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'now', return_value=dt)
    def test_repr(self, mck_now):

        self.submit.update_history('DOne!')
        got_repr = repr(self.submit)
        expect_repr = "Name: dfols_r Nmodels:3 Status: CREATED: 3 Model_Types:myModel: 3 Last changed at 2022-01-01 00:00:00"
        self.assertEqual(got_repr, expect_repr)

    def test_models_to_submit(self):
        # test models_to_submit works. Should get models
        # with status INSTANTIATE or CONTINUE

        # first case should be empty
        self.assertEqual(self.submit.models_to_submit(), [])

        # instantiate all models.
        self.submit.instantiate()
        self.assertEqual(len(self.submit.models_to_submit()), 3)
        # mark 1 for continue
        list(self.submit.model_index.values())[0].status = 'CONTINUE'
        self.assertEqual(len(self.submit.models_to_submit()), 3)
        # mark them all as running
        for m in self.submit.model_index.values():
            m.status = 'RUNNING'
        self.assertEqual(len(self.submit.models_to_submit()), 0)

    def test_to_dict(self):
        # test to_dict method.
        study_dict = self.submit.to_dict()
        expected_dict = vars(self.submit)
        # now replace models!
        expected_dict['model_index'] = {k: m.config_path for k, m in expected_dict['model_index'].items()}
        # and evil hack for config
        expected_dict['config'] = vars(expected_dict['config'])
        # and engine
        expected_dict['engine'] = engine.sge_engine()
        self.assertEqual(study_dict, expected_dict)

    def test_iterations(self):
        # test iterations command works.
        iters = self.submit.iterations()
        # iters should be a 1 element list
        self.assertEqual(len(iters), 1)
        self.assertEqual(len(iters[0]),3) # and 3 models.
        # now add a model
        pDict= iters[0][-1].parameters
        pDict.update(dict(VF1=pDict.get("VF1",2.0)*1.1))
        new_model= self.submit.create_model(pDict)
        self.submit.update_iter([new_model])
        iters = self.submit.iterations()
        # iters should be a 2 element list
        self.assertEqual(len(iters), 2)
        self.assertEqual(len(iters[1]),1) # and 1 models.
        self.assertEqual(iters[1][0],new_model)

    def test_guess_failed(self):
        # test guess_failed
        # 1) Setup up Submit object with models. Then set their status to Running with the ids set.
        submit = self.submit
        submit.instantiate() # instantiate all the models
        # then set status to RUNNING and set up model_jids
        jid='123456'
        for model in submit.model_index.values():
            model.status='RUNNING'
            model.model_jids.append(jid)
            jid=str(int(jid)+1)

        with unittest.mock.patch('engine.sge_engine.job_status',autospec=True,return_value='notFound') as mck:
            failed_models = submit.guess_failed()
            self.assertEqual(len(failed_models),3)
            for model in failed_models:
                self.assertEqual(model.status,"FAILED")

        # now try again but mocking job_status to Queuing. Status of model should be RUNNING
        for model in submit.model_index.values():
            model.status='RUNNING'

        with unittest.mock.patch('engine.sge_engine.job_status',autospec=True,return_value='Queuing') as mck:
            failed_models = submit.guess_failed()
            self.assertEqual(len(failed_models),0)
            for model in submit.model_index.values():
                self.assertEqual(model.status,"RUNNING")

    def test_running_models(self):
        # test running_models works
        submit = self.submit
        # nothing is running so expect no running models
        rmodels = submit.running_models()
        self.assertEqual(len(rmodels),0)
        # now make models running!
        submit.instantiate()
        # set the status to RUNNING
        for model in submit.model_index.values():
            model.status = "RUNNING"
        rmodels = submit.running_models()
        self.assertEqual(rmodels,list(submit.model_index.values()))



    """
    AI PROMPT/Spec:
    Write unit tests for the `process` method of the `SubmitStudy` class.
    Tests should use unittest for testing. 
      In setup it should create a SubmitStudy object with a model_index and do the following to set up self.submit:
        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(self.tmpDir.name)
        optclim3 = Model.expand('$OPTCLIMTOP/OptClimVn3/')
        refDir = optclim3/'configurations/example_Model'
        cpth = refDir/"configurations/dfols14param_opt3.json"
        refDir = refDir/'reference'
        config = StudyConfig.readConfig(cpth)
        config.baseRunID('ZZ')

        self.submit = SubmitStudy.SubmitStudy(config, model_name='Model', rootDir=testDir,next_iter_cmd=['run myself'])
    It should add five models (submit.create_model does that)
    Three of those should have status set to 'SUCCEEDED',  one set to 'PROCESSED', and another set to 'RUNNING'. 
    All should have pp_jid set to an arbitrary 6 digit string.
    These will require directly modifying the models which live in submit.model_index.

    
    It should mock:
         Model.pp_job_status -- see test descriptions for what this should return.
         Model.process to run Model.set_status and then return a nonsense string. 
           Note that set_status can fail and the mock should just let that happen if it does.
        submit.dump_config to not do anything.
    
    mocks likely need to be done before models are created. But say if not so.
    Tests should do the following:
    1) Post-processing not running    
       mocked pp_job_status returns 'notFound' for all models. 
       returns a list of models that were processed. 
          They should be the first three models and have status changed from 'SUCCEEDED' to 'PROCESSED'.
       the Model.process method was called 3 times.
       dump_config was called once.

    2) Post-processing running
       mocked pp_job_status returns 'Running' for all models.
       Model.process method was not called
       A empty list  should be returned.
       dump_config was not called.
    3) Mixed case
       1 of the Model.job_status returns 'Running', the rest return 'notFound'
       Model.process was called twice.
       returns 2 models that were processed (the two that returned 'NotFound') with status changed to 'PROCESSED'.
       dump_config was called once.
    4) One of the Model.pp_job_status returns None. Rest job_status returns 'Running'
       Model.process was called once
       returns a list with the model that had no pp_jid.
       dump_config was called once.

    """

class test_Process(unittest.TestCase):
    def setUp(self):
        def process_side_effect(self,*args, **kwargs):
            # Set status to PROCESSED if possible

            self.set_status('PROCESSED')
            return "nonsense"

        def pp_job_status_side_effect(self, *args, **kwargs):
            # Return value of pp_job_status or 'notFound' if pp_job_status is not set
            status = getattr(self, 'pp_jid_status', 'notFound')
            return status
        # Patch Model.pp_job_status and Model.process before model creation
        self.pp_job_status_patcher = unittest.mock.patch.object(Model, 'pp_job_status',autospec=True,side_effect=pp_job_status_side_effect)
        self.dump_config_patcher = unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'dump_config',autospec=True,)
        self.process_patcher = unittest.mock.patch.object(Model, 'process', autospec=True,side_effect=process_side_effect)

        self.mock_pp_job_status = self.pp_job_status_patcher.start()
        self.mock_dump_config = self.dump_config_patcher.start()
        self.mock_process = self.process_patcher.start()


        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(self.tmpDir.name)
        optclim3 = Model.expand('$OPTCLIMTOP/OptClimVn3/')
        refDir = optclim3 / 'configurations/example_Model'
        cpth = refDir / "configurations/dfols14param_opt3.json"
        config = StudyConfig.readConfig(cpth)
        config.baseRunID('ZZ')

        self.submit = SubmitStudy.SubmitStudy(config, model_name='Model', rootDir=testDir,
                                              next_iter_cmd=['run myseld'])

        # Add 5 models to the submit object
        self.models=[] # 'copy' of models in submit.model_index
        for i in range(5):
            params = dict(indx=i, VF1=2.0 + i * 0.1, CT=1e-4 + i * 1e-5)
            self.models.append(self.submit.create_model(params))


        # Set statuses and pp_jid
        for i, model in enumerate(self.models):
            if i < 3:
                model.status='SUCCEEDED'
            elif i == 3:
                model.status='PROCESSED'
            else:
                model.status='RUNNING'
            model.pp_jid = f'{100000 + i}'

    def tearDown(self):
        self.pp_job_status_patcher.stop()
        self.process_patcher.stop()
        self.dump_config_patcher.stop()
        self.tmpDir.cleanup()


    def test_post_processing_not_running(self):
        # All pp_job_status return 'notFound'
        #self.mock_pp_job_status.return_value = 'notFound'

        self.mock_dump_config.reset_mock()
        self.mock_process.reset_mock()
        expected_status = ['PROCESSED'] * 3
        processed=self.submit.process()
        self.assertEqual([p.status for p in processed],expected_status)  # Get statuses of processed models

        self.assertEqual(self.mock_process.call_count, 3)
        self.mock_dump_config.assert_called_once()

    def test_post_processing_running(self):
        # All pp_job_status return 'Running'
        for model in self.models:
            model.pp_jid_status = 'Running'
        self.mock_process.reset_mock()
        self.mock_dump_config.reset_mock()

        processed = self.submit.process()
        self.assertEqual(processed, [])
        self.assertEqual(self.mock_process.call_count, 0)
        self.mock_dump_config.assert_not_called()

    def test_mixed_case(self):
        # First model returns 'Running', rest 'motFound'
        self.models[0].pp_jid_status = 'Running'

        self.mock_dump_config.reset_mock()
        expected_status = ['PROCESSED'] * 2



        processed = self.submit.process()
        # Only models 1 and 2 should be processed
        self.assertEqual([p.status for p in processed], expected_status)
        self.assertEqual(self.mock_process.call_count, 2)
        self.mock_dump_config.assert_called_once()

    def test_one_pp_job_status_none(self):
        # First model returns None, rest 'Running'

        self.models[0].pp_jid_status = None
        for model in self.models[1:]:
            model.pp_jid_status = 'Running'
        self.mock_dump_config.reset_mock()
        expected = ['PROCESSED']
        processed = self.submit.process()
        self.assertEqual([p.status for p in processed], expected)
        self.assertEqual(self.mock_process.call_count, 1)
        self.mock_dump_config.assert_called_once()


"""
AI Prompt/Spec:
Write unit tests for the `kill` method of the `SubmitStudy` class. Use unittest for testing.
mocks needed to  avoid subprocess.check_output calls and use of slurm/sge for testing.
SubmitStudy.resub_status.  Should return value of SubmitStudy.resub_job_status if set otherwise 'notFound'.
SubmitStudy.run_cmd should return 'nonsense' string.
Model.kill -- should return [Model.pp_jid, Model.model_jids[-1]] Model.kill() will be tested in the Model tests.
All mocks should use autospec=True.

Setup should create a SubmitStudy object with a model_index and do the following to set up self.submit use part of the setUp from test_process. 

Following tests should be done:
1) Run submit.kill() with resub_status set to None -- should return a 10 element list from the model.pp_jid and model.model_jids[-1] values. model.kill should be called 5 times.
2) Set  submit.resub_status to 'Running' -- should return an 11 element list from the model.pp_jid and model.model_jids[-1] values + submit.resub_jids[-1]. model.kill should be called 5 times.
3) Make model.kill() return an empty list.  And run test#2 should get a 1 element list with the resub_jids[-1] value. model.kill should be called 5 times.
"""


def model_kill(self, *args, **kwargs):
    # Return pp_jid and last model_jid
    result = []
    if self.pp_jid is not None:
        result.append(self.pp_jid)
    if self.model_jids:
        result.append(self.model_jids[-1])
    return result


def resub_status_side_effect(self, *args, **kwargs):
    # Return value of pp_job_status or 'notFound' if pp_job_status is not set
    status = getattr(self, 'resub_job_status', 'notFound')
    return status

@unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'run_cmd', autospec=True, return_value='nonsense')
@unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'resub_status', autospec=True,side_effect=resub_status_side_effect)
@unittest.mock.patch.object(Model, 'kill', autospec=True,side_effect=model_kill)
class test_KILL(unittest.TestCase):
    def setUp(self):

        # Patch Model.pp_job_status and Model.process before model creation
        #self.resub_status_patcher = unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'resub_status',autospec=True)




        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(self.tmpDir.name)
        optclim3 = Model.expand('$OPTCLIMTOP/OptClimVn3/')
        refDir = optclim3 / 'configurations/example_Model'
        cpth = refDir / "configurations/dfols14param_opt3.json"
        config = StudyConfig.readConfig(cpth)
        config.baseRunID('ZZ')

        self.submit = SubmitStudy.SubmitStudy(config, model_name='Model', rootDir=testDir,
                                              next_iter_cmd=['run myseld'])

        # Add 5 models to the submit object
        self.models=[] # 'copy' of models in submit.model_index
        for i in range(5):
            params = dict(indx=i, VF1=2.0 + i * 0.1, CT=1e-4 + i * 1e-5)
            self.models.append(self.submit.create_model(params))


        # Set statuses and pp_jid
        for i, model in enumerate(self.models):
            if i < 3:
                model.status='SUCCEEDED'
            elif i == 3:
                model.status='PROCESSED'
            else:
                model.status='RUNNING'
            model.pp_jid = f'{100000 + i}'
            model.model_jids = [f'{200000 + i}']

        self.submit.next_iter_jids=['99999']

    def tearDown(self):

        self.tmpDir.cleanup()


    def test_kill_no_resub_status(self,mock_model_kill, mock_resub_status,mock_run_cmd):
        # No resub_status set, should return 10 elements from model.pp_jid and model.model_jids[-1]


        result = self.submit.kill()
        expected_jids=[]
        for model in self.models:
            expected_jids.append(model.pp_jid)
            expected_jids.append(model.model_jids[-1])


        self.assertEqual(result, expected_jids)
        self.assertEqual(mock_model_kill.call_count, 5)
        self.assertEqual(mock_run_cmd.call_count, 0)

    def test_kill_with_resub_status(self,mock_model_kill, mock_resub_status,mock_run_cmd):
        # Set resub_status to 'Running', should return 11 elements from model.pp_jid, model.model_jids[-1] and submit.resub_jids[-1]
        self.submit.resub_job_status = 'Running'


        result = self.submit.kill()
        expected_jids=[]
        for model in self.models:
            expected_jids.append(model.pp_jid)
            expected_jids.append(model.model_jids[-1])
        expected_jids.append(self.submit.next_iter_jids[-1])
        self.assertEqual(result, expected_jids)
        self.assertEqual(mock_model_kill.call_count, 5)
        self.assertEqual(mock_run_cmd.call_count, 1)

    def test_kill_with_empty_model_kill(self, mock_model_kill, mock_resub_status, mock_run_cmd):
        # Make model.kill() return an empty list for all models
        mock_model_kill.side_effect = lambda self, *a, **k: []
        self.submit.resub_job_status = 'Running'

        result = self.submit.kill()
        expected_jids = [self.submit.next_iter_jids[-1]]
        self.assertEqual(result, expected_jids)
        self.assertEqual(mock_model_kill.call_count, 5)
        self.assertEqual(mock_run_cmd.call_count, 1)

if __name__ == '__main__':
    unittest.main()
