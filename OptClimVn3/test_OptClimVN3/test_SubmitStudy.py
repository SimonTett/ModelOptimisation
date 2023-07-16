"""
Test cases for SubmitStudy classes
"""
import datetime
import importlib.resources
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


def gen_time():
    # used to mock Model.now()
    time = datetime.datetime(2000, 1, 11, 0, 0, 0)
    timedelta = datetime.timedelta(seconds=1)
    while True:
        time += timedelta
        yield time


class myModel(Model):
    pass


# class that inherits from Model.,
times = gen_time()
traverse = importlib.resources.files("Models")
with importlib.resources.as_file(traverse.joinpath("parameter_config/example_Parameters.csv")) as pth:
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
        submit = SubmitStudy.SubmitStudy(config, model_name='myModel', rootDir=testDir)
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
        params = dict(VF1=2.2, RHCRIT=3)
        model = self.submit.create_model(params, dump=False)
        self.assertTrue(isinstance(model, Model))
        self.assertEqual(model.parameters, params)
        # now create one that goes to disk,
        params.update(VF1=2.1)
        model2 = self.submit.create_model(params)
        self.submit.dump_config(dump_models=True)
        # as dump_models is True expect SubmitStudy obj on disk and model as well.
        # load them and compare.
        m2 = Model.load_model(model.config_path)
        m3 = Model.load_model(model2.config_path)
        self.assertEqual(model, m2)
        self.assertEqual(model2, m3)
        sub2 = self.submit.load(self.submit.config_path)
        self.assertEqual(self.submit, sub2)

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
        job_nos = range(34567,34567+2*len(submit.model_index)) # job numbers
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
            self.assertEqual(mck_output.call_count, 6)
            # expect that the ob id is every 2nd job_no (as a str)
            # so lets check that.
            for model,jno in zip(submit.model_index.values(),job_nos[0::2]):
                self.assertEqual(model.pp_jid,str(jno))

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
            for indx,model in enumerate(submit.model_index.values()):
                self.assertEqual(model.status, "SUBMITTED")
                self.assertEqual(model.pp_jid, str(job_nos[indx*2]))
                mj = job_nos[indx*2+1] # jid for start job. Continue job should have this +100
                self.assertEqual(model.model_jids,[str(mj),str(mj+100)])

        # final actual run test. Next job is submitted.
        output ='Next Submitted 345679'
        with unittest.mock.patch("subprocess.check_output",
                                 autospec=True, return_value=output) as mck_output:
            for m in submit.model_index.values():
                m.status = 'INSTANTIATED'
                m.pp_jid = None # set post_process_cmd back to None.
                m.model_jids = [] # and model list to empty
            submit.submit_all_models(next_iter_cmd=['run myself'])
            self.assertEqual(mck_output.call_count, 7)  # 7 cases. 3 x (model + pp submit) + one next job.

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
                sim_obs[oname] = v ** 2
                obs_count += 1
            return pd.Series(sim_obs)

        with unittest.mock.patch("subprocess.check_output",
                                 autospec=True, return_value="some value 345678") as mck_output:
            submit = copy.deepcopy(self.submit)
            submit.instantiate()  # instantiate all models.
            submit.submit_all_models(next_iter_cmd=['run myself'], fake_fn=fake_function)
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

if __name__ == '__main__':
    unittest.main()
