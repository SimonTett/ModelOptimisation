import json
import logging
import platform
import sys
import unittest

import genericLib
import unittest.mock
import SubmitStudy
import tempfile
import pathlib
import engine
from simple_model_pars_json import simple_model_pars as simple_model
import StudyConfig
import shutil
import re
import subprocess
import os



"""
Test simple_model_pars_json.
"""

genericLib.setup_env()
my_logger = genericLib.setup_logging(
    level=logging.WARNING
)
class Test_simple_model_pars_json(unittest.TestCase):
    def setUp(self) -> None:
        # set up a SubmitStudy as needed.
        self.tmpDir = tempfile.TemporaryDirectory()
        self.tmpDir2 = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(self.tmpDir.name)
        testDir2 = pathlib.Path(self.tmpDir2.name)
        optclim3 = genericLib.expand('$OPTCLIMTOP/OptClimVn3/')
        refDir = optclim3/'configurations/example_simple_model_pars_json/reference'
        self.refDir = refDir
        cpth = optclim3/"configurations/dfols14param_opt3.json"
        config = StudyConfig.readConfig(cpth)
        config.baseRunID('ZZ')
        submit = SubmitStudy.SubmitStudy(config, model_name='simple_model', rootDir=testDir)
        self.submit = submit
        self.testDir = testDir
        self.testDir2 = testDir2

        # set up another model properly
        eng = engine.abstractEngine.create_engine('SGE')
        params = dict(VF1=2.3,RHCRIT=0.5)
        model = simple_model('model002',reference=self.refDir,engine=eng,
                                          model_dir=self.submit.rootDir/'model002',
                                          study=self.submit.to_study(),parameters=params)
        self.model = model


    def tearDown(self) -> None:
        # clean up a bit.
        self.tmpDir.cleanup()
        self.tmpDir2.cleanup()



    def test_init(self):
        """
        Test init by creating a model and verify that StudyConfig_path attribute is as expected.
        :return:
        """
        model = simple_model('model001',reference=self.refDir,
                                          model_dir=self.submit.rootDir/'model001',
                                          study=self.submit.to_study())
        self.assertEqual(model.StudyConfig_path,self.submit.config.fileName())
        model = simple_model('model002',reference=self.refDir,
                                          model_dir=self.submit.rootDir/'model002')
        self.assertIsNone(model.StudyConfig_path)


    def test_modify_model(self):
        # test that if we modify the model that script is as expected. Just counts the modify lines -- expect 3.
        modifyStr = '## modified *$'
        model = simple_model('model002',reference=self.refDir,
                                          model_dir=self.submit.rootDir/'model002') # create a model.
        # copy the model info across.
        shutil.copytree(self.refDir,model.model_dir)
        # now modify it and test script
        model.modify_model()
        count = 0
        # count the number of modifystr there are.
        with open(model.model_dir/'run_simple_model_pars_json.py', 'r') as f:
            for line in f:
                if re.search(modifyStr, line): count += 1

        # expect 19 modifications -- import, 5*(start, fail, continue)+3 (fail has some extra cmds)
        self.assertEqual(count,19)


    def test_set_params(self):
        params = self.model.parameters
        model = simple_model('model001',reference=self.refDir,
                                        run_info=dict(submit_engine='SGE'),
                                        model_dir=self.submit.rootDir/'model001',
                                        study=self.submit.to_study(),parameters=params)

        shutil.copytree(self.refDir,model.model_dir)
        model.set_params(params)
        set_params = {p:model.read_param(p) for p in params.keys()}
        self.assertEqual(params,set_params)

        model.instantiate() # instantiate it -- which should set the params
        set_params = {p:model.read_param(p) for p in params.keys()}
        self.assertEqual(params,set_params)



    def test_submit_cmd(self):
        # test that submit_cmd is as expected.
        self.model.instantiate() # create the model on disk.
        run_info=dict()
        eng = engine.abstractEngine.create_engine('SGE') # SGE cmd
        cmd = self.model.submit_cmd()
        self.assertIsInstance(cmd,list)
        model = self.model
        outdir = model.model_dir / 'model_output'
        expected_cmd = eng.submit_cmd([model.model_dir/model.submit_script,str(model.StudyConfig_path)],
                                      f"{model.name}{len(model.model_jids):05d}", outdir, rundir=model.model_dir,time=30)
        self.assertEqual(cmd,expected_cmd)

    def test_dump_load(self):
        # test that dumping and loading a model works.
        m = self.model
        m.instantiate()
        m.dump_model()
        m2 = m.load_model(m.config_path)
        m2 == m
        self.assertEqual(m,m2)

    def test_run_simple_model(self):
        """ Actually test running the model works."""
        model = self.model
        model.parameters.update(sleep_time=1)
        model.instantiate()
        model.set_status("SUBMITTED") # make state submittable.
        cmd = ["./"+str(model.submit_script),str(model.StudyConfig_path)]
        if platform.system() == 'Windows':
            #cmd = [sys.executable,f'{model.submit_script}',f'{model.StudyConfig_path}']
            cmd.insert(0,sys.executable)
        #else:
            #cmd = ["./"+str(model.submit_script),str(model.StudyConfig_path)]

        # fake env so ID can be found.
        vars = ['JOB_ID','SLURM_JOB_ID']
        for v in vars:
            os.environ[v]= '123456'
        result=subprocess.run(cmd,cwd=model.model_dir,check=True,text=True)
        # on linux note that shell=True requires a string to be passed not a 
        # list.
        model2 = simple_model.load_model(model.config_path)
        self.assertEqual(model2.status,'SUCCEEDED')
        self.assertEqual(model2.model_jids,['123456'])

    def test_gen_params(self):
        """
        Test cases:
        - Test with no params. Should get appropriate values back
        - Test with `parameters` containing valid data.
        - Test to ensure `ValueError` is raised for duplicate namelist.
        """
        model = self.model
        shutil.copytree(self.refDir,model.model_dir)
        result = model.gen_params()
        pars = model.parameters.copy()
        pars.update(model.parameters_no_key)
        self.assertEqual(len(result),len(pars))
        for key,v in pars.items():
            nl,v2 = model.param_info.param(model,key,v)[0]
            self.assertEqual(result[nl],v2)

        # Specify some parameters
        params = dict(VF1=3.0,RHCRIT=0.6)
        result = model.gen_params(params)
        self.assertEqual(len(result),len(params))
        for key,v in params.items():
            nl,v2 = model.param_info.param(model,key,v)[0]
            self.assertEqual(result[nl],v2)

        # test for duplicate namelist.
        # that means hacking param_info to have a duplicate namelist.
        model.param_info.param_constructors['VF2']=model.param_info.param_constructors['VF1']
        # Could do this properly using register but that stops duplicate nl.


        with self.assertRaises(ValueError):
            model.gen_params(dict(VF1=3.0,RHCRIT=0.6,VF2=3.1))



    def test_read_param(self):
        """
        Test cases:
        - Test with a RHCRIT  that is callable and get back
        - Test with VF1 that is a `NamelistVar`.
        - Test with an invalid `parameter` to ensure `KeyError` is raised.
        """

        self.model.instantiate()
        with (self.model.model_dir/'parameters.json').open('r+t') as f:
            param_values = json.load(f)
        rhcrit = self.model.read_param('RHCRIT')
        self.assertEqual(rhcrit,param_values["model_params"]['RHCRIT'][0])
        vf1 = self.model.read_param('VF1')
        self.assertEqual(vf1,param_values["model_params"]['VF1'])
        with self.assertRaises(KeyError):
            self.model.read_param('INVALID')

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
        expected_nl = self.model.param_info.param_constructors['VF1'][0]
        expected_val = 2.3
        (nl,val) = self.model.param('VF1',expected_val)[0]
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
        with self.assertRaises(ValueError):
            self.model.param('BAD',0.5)

        self.assertEqual(self.model.param('NONE',0.5),[])




if __name__ == '__main__':
    unittest.main()
