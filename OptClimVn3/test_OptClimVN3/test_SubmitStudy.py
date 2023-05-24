"""
Test cases for SubmitStudy classes
"""
import datetime
import importlib.resources
import pathlib
import tempfile
import unittest.mock  # need to mock the run case.
import unittest
import StudyConfig
import SubmitStudy
from Model import Model
import copy


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
times=gen_time()
traverse = importlib.resources.files("Models")
with importlib.resources.as_file(traverse.joinpath("parameter_config/example_Parameters.csv")) as pth:
    myModel.update_from_file(pth)
class MyTestCase(unittest.TestCase):

    @unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'now', side_effect=gen_time()) # regen times every time!
    def setUp(self,mck_now):
        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(self.tmpDir.name)
        cpth = SubmitStudy.SubmitStudy.expand("$OPTCLIMTOP/OptClimVn3/configurations/dfols14param_opt3.json")
        config = StudyConfig.readConfig(cpth)
        config.baseRunID('ZZ')
        submit = SubmitStudy.SubmitStudy(config,model_name='myModel',rootDir=testDir)
        # create some models
        for param in [dict(VF1=3,CT=1e-4),dict(VF1=2.4,CT=1e-4),dict(VF1=2.6,CT=1e-4)]:
            submit.create_model(param,dump=True)
        self.submit=submit
        self.testDir = testDir

    def tearDown(self):
        self.tmpDir.cleanup()

    # first tests for engines
    def test_eng_setup_engine(self):
        # get some thing sensible..
        cmd = ["echo"], ['fred'] # cmd that will be ran
        jid = "203675"
        for name in ['SGE','SLURM']:
            eng = SubmitStudy.engine.setup_engine(name)
            # should be name and, except for engine_name, be functions
            for k,v in vars(eng).items():
                if k == "engine_name":
                    self.assertEqual(v,name)
                else:
                    self.assertTrue(callable(v))
                    #run the command -- should be a list but the arguments depend on what it is!
                    if k == "submit_fn":
                        result = v(cmd,'tst_sub',pathlib.Path.cwd())
                        self.assertIsInstance(result,list)
                    elif k == "array_fn":
                        result = v(cmd,'tst_sub',pathlib.Path.cwd(),10)
                        self.assertIsInstance(result,list)
                    elif k in ['release_fn','kill_fn']:
                        result = v(jid)
                        self.assertIsInstance(result,list)
                    elif k in ['jid_fn']:
                        jid = "56745"
                        if name == 'SGE':
                            result = v("cmd name "+jid)
                            self.assertEqual(jid, result)
                        elif name == 'SLURM':
                            with self.assertRaises(NotImplementedError):
                                result = v("cmd name "+jid)
                        else:
                            raise NotImplementedError(f" implement test for {name} jid_fn")
                    else:
                        raise NotImplementedError(f"Do not know how to test {k}")

        # test unknown engin causes failure
        with self.assertRaises(ValueError):
            SubmitStudy.engine.setup_engine('fred')

    def test_eng_to_dict(self):
        # test that the dct we get makes sense. It should be the function *names* and engine_name
        eng = SubmitStudy.engine.setup_engine('SLURM')
        dct = eng.to_dict()
        for k,v in dct.items():
            if k == 'engine_name':
                self.assertEqual(v,eng.engine_name)
            else:
                eng_v = getattr(eng,k)
                self.assertIsInstance(v,str)
                self.assertTrue(callable(eng_v))
                self.assertEqual(v,eng_v.__name__)


    def test_eng_from_dict(self):
        # test that can convert a dict -- functions get generated from engine_name
        dct = dict(submit_fn = 'sge_submit_fn',array_fn='sge_array_fn',
                   release_fn='sge_release_fn',kill_fn='sge_kill_fn',
                   jid_fn='sge_jid_fn',
                   engine_name='SGE')
        eng = SubmitStudy.engine.from_dict(dct)
        self.assertEqual(eng.to_dict(),dct)
        # fail cases. See engine_name to SLURM. Should fail with the existing names.
        dct.update(engine_name='SLURM')
        with self.assertRaises(ValueError):
            eng = SubmitStudy.engine.from_dict(dct)
        # fix all names and should run.
        for key in dct.keys():
            dct[key] = dct[key].replace("sge","slurm")
        eng = SubmitStudy.engine.from_dict(dct)
        self.assertEqual(eng.to_dict(),dct)


    def test_create_model(self):
        """
        Test that can create a model.
        :return:
        """
        params=dict(VF1=2.2,RHCRIT=3)
        model = self.submit.create_model(params,dump=False)
        self.assertTrue(isinstance(model,Model))
        self.assertEqual(model.parameters,params)
        # now create one that goes to disk,
        params.update(VF1=2.1)
        model2 = self.submit.create_model(params)
        # as dump is True expect SubmitStudy obj on disk as well as model.
        # load them and compare.
        m2 = Model.load_model(model.config_path)
        m3 = Model.load_model(model2.config_path)
        self.assertEqual(model,m2)
        self.assertEqual(model2,m3)
        sub2 = self.submit.load(self.submit.config_path)
        self.assertEqual(self.submit, sub2)



    @unittest.mock.patch.object(SubmitStudy.SubmitStudy,'now',side_effect=times)
    def test_delete(self,mck_now):
        # can we delete things.
        pth = self.submit.config_path
        self.submit.config.baseRunID(value='ZZ')
        mpths = [m.config_path for m in self.submit.model_index.values()]
        if not pth.exists():
            raise ValueError(f"config {pth} does not exist")
        for mpth in mpths:
            if not mpth.exists():
                raise ValueError(f"Model config {mpth} does not exist" )
        nhist = len(self.submit.history)
        self.submit.delete() # should delete everything including models.
        # models should all be gone, no model index and next_name be ZZ000
        self.assertFalse(pth.exists())
        for mpth in mpths:
            self.assertFalse(mpth.exists())
        self.assertEqual(self.submit.gen_name(),'ZZ000')
        self.assertEqual(len(self.submit.history),nhist+1) # added deleted



    def test_dump_load(self):
        submit = self.submit

        submit.dump(submit.config_path)
        nsub = SubmitStudy.SubmitStudy.load(submit.config_path)
        self.assertEqual(submit,nsub) # should be identical


    def test_gen_name(self):
        # have generated three models so gen_model should be
        self.submit.name_values = None
        name = self.submit.gen_name()
        self.assertEqual(name,'ZZ000')
        # set name_values[0] to 11 (should get a)
        self.submit.name_values[0]=9
        name = self.submit.gen_name()
        self.assertEqual(name,'ZZ00a')
        # and increase it to 36
        self.submit.name_values = [35,0,0]
        name = self.submit.gen_name()
        self.assertEqual(name, 'ZZ010')

        self.submit.name_values = [34,35,35]
        name = self.submit.gen_name()
        self.assertEqual(name, 'ZZzzz')



    def test_instantiate(self):
        # test we can instantiate all relevant models.
        # will instantiate  one model directly and then instantiate everything.
        # should have two new models then!
        submit = self.submit
        lst_model = list(submit.model_index.values())[-1]
        lst_model.instantiate()
        # at this point expect model_dir to exist and be in rootDir (where the config is)
        self.assertTrue(lst_model.model_dir.exists())
        self.assertTrue(lst_model.model_dir.is_relative_to(submit.rootDir))
        # only contents at this point should be the study config, the three model configs (which go in rootDir)
        # and one modelDir.

        pths_expected = [m.config_path for m in submit.model_index.values()]
        pths_expected += [submit.config_path,lst_model.model_dir]
        for p in pths_expected:
            self.assertTrue(p.exists())
        self.assertTrue(lst_model.model_dir.is_dir())

        # and rootdir should contain ONLY pths_expected.
        pths_got = set(submit.rootDir.glob("*"))
        self.assertEqual(set(pths_got),set(pths_expected))

        # now instantiate. Should have two more directories
        submit.instantiate()
        pths_expected += [m.model_dir for m in submit.model_index.values()]
        for p in pths_expected:
            self.assertTrue(p.exists())
        for m in submit.model_index.values():
            self.assertTrue(m.model_dir.is_dir())
        # and rootdir should contain ONLY pths_expected.
        pths_got = set(submit.rootDir.glob("*"))
        self.assertEqual(set(pths_got), set(pths_expected))

    # need to mock both SubmitStudy and myModel now.
    @unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'now', side_effect=times)
    @unittest.mock.patch.object(myModel,'now',side_effect=times)
    def test_submit_all_models(self,mck_now,mck_model_now):

        # set up the fake rtn output
        submit = copy.deepcopy(self.submit)

        rtn_job_output = ['postprocess submitted 345678.10']  # array of pp jobs as no next job wanted
        model_output = ['Model submitted']*len(submit.model_index)
        output = rtn_job_output + model_output
        with unittest.mock.patch("subprocess.check_output",
                                 autospec=True, side_effect=output) as mck_output:
            submit.submit_all_models()
            # run submit -- nothing should happen as no models are instantiated
            mck_output.assert_not_called()
            submit.instantiate() # instantiate all models.
            submit.submit_all_models()
            # run submit -- should submit the pp process and three models. so 4 times
            self.assertEqual(mck_output.call_count,4)
            # expect to have a tempConfigList.txt file -- with 3 lines as model config file
            tempFile=submit.rootDir/'tempConfigList.txt'
            expected_output=[str(model.config_path) for model in submit.model_index.values()]
            with open(tempFile,'rt') as f:
                output= f.readlines()
            output = [o.rstrip("\n") for o in output]
            self.assertEqual(output,expected_output)
            # expect that the post process cmd is ['qrls',jid]
            # so lets check that.
            for indx,model in enumerate(submit.model_index.values()):
                self.assertEqual(model.post_process_cmd,['qrls',f"345678.{indx+1}"])

        # now have the models all fail and then continue.  Should only have three cases -- the models
        # All status should be submitted and the post-process cmd should be unchanged
        with unittest.mock.patch("subprocess.check_output",
                                 autospec=True, side_effect=model_output) as mck_output:
            for model in submit.model_index.values():
                model.status="CONTINUE"
            submit.submit_all_models()
            self.assertEqual(mck_output.call_count,3) # three cases
            # check models are as expected.
            for indx,model in enumerate(submit.model_index.values()):
                self.assertEqual(model.status,"SUBMITTED")
                self.assertEqual(model.post_process_cmd,['qrls',f"345678.{indx+1}"])

        # final actual run test. Next job is submitted.
        output = rtn_job_output + ['Next Submitted 345679']+ model_output
        with unittest.mock.patch("subprocess.check_output",
                                 autospec=True, side_effect=output) as mck_output:
            for m in submit.model_index.values():
                m.status='INSTANTIATED'
            submit.submit_all_models(next_iter_cmd=['run myself'])
            self.assertEqual(mck_output.call_count, 5)  # 5 cases

        # final tests -- history and output as expected.
        # expect 13 history:
        #    1 start, 3 x model created, 3 x instantiated, pp job submitted, models submitted,
        #      models continued, pp job submitted, next job submitted, models submitted.
        #
        # and 4+3+5 outputs
        self.assertEqual(len(submit.history),13)
        self.assertEqual(len(submit.output),3) # pp ran twice plus next iter.

        # now fake it. subprocess.check_output should not run anything.
        import pandas as pd
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
            submit.submit_all_models(next_iter_cmd=['run myself'],fake_fn=fake_function)
            mck_output.assert_not_called()

    dt=datetime.datetime(2022,1,1,0,0,0)
    @unittest.mock.patch.object(SubmitStudy.SubmitStudy, 'now', return_value=dt)
    def test_repr(self,mck_now):

        self.submit.update_history('DOne!')
        got_repr = repr(self.submit)
        expect_repr="Name: dfols_r Nmodels:3 Status: CREATED: 3 Model_Types:myModel: 3 Last changed at 2022-01-01 00:00:00"
        self.assertEqual(got_repr,expect_repr)

    def test_models_to_submit(self):
        # test models_to_submit works. Should get models
        # with status INSTANTIATE or CONTINUE

        # first case should be empty
        self.assertEqual(self.submit.models_to_submit(),[])

        # instantiate all models.
        self.submit.instantiate()
        self.assertEqual(len(self.submit.models_to_submit()),3)
        # mark 1 for continue
        list(self.submit.model_index.values())[0].status='CONTINUE'
        self.assertEqual(len(self.submit.models_to_submit()), 3)
        # mark them all as running
        for m in self.submit.model_index.values():
            m.status='RUNNING'
        self.assertEqual(len(self.submit.models_to_submit()), 0)





if __name__ == '__main__':
    unittest.main()
