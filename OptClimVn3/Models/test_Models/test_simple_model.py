import json
import platform
import unittest
import generic_json
import unittest.mock
import SubmitStudy
import tempfile
import pathlib
import engine
from Models import *
import StudyConfig
import shutil
import re
import subprocess
import os
"""
Test simple_model.
"""


class Test_simple_model(unittest.TestCase):
    def setUp(self) -> None:
        # set up a SubmitStudy as needed.
        self.tmpDir = tempfile.TemporaryDirectory()
        self.tmpDir2 = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(self.tmpDir.name)
        testDir2 = pathlib.Path(self.tmpDir2.name)
        optclim3 = simple_model.expand('$OPTCLIMTOP/OptClimVn3/')
        refDir = optclim3/'configurations/example_simple_model/reference'
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
        params = dict(vf1=2.3,rhcrit=0.5)
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
        with open(model.model_dir/'run_simple_model.py', 'r') as f:
            for line in f:
                if re.search(modifyStr, line): count += 1

        # expect sevent modifications -- import, 2xstart, 5xfail and 2succeeded.
        self.assertEqual(count,10)


    def test_set_params(self):
        params = self.model.parameters
        model = simple_model('model001',reference=self.refDir,
                                        run_info=dict(submit_engine='SGE'),
                                        model_dir=self.submit.rootDir/'model001',
                                        study=self.submit.to_study(),parameters=params)
        model.model_dir.mkdir(exist_ok=True,parents=True)
        model.set_params(params)
        with open(model.model_dir/'params.json','r') as fp:
            set_params = json.load(fp)
        self.assertEqual(params,set_params)

        model.instantiate() # instantiate it
        with open(model.model_dir/'params.json','r') as fp:
            set_params = json.load(fp)
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
        expected_cmd = eng.submit_cmd([model.submit_script,str(model.StudyConfig_path)],
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
        if platform.system() == 'Windows':
            cmd = ['python',f'{model.submit_script}',f'{model.StudyConfig_path}']
        else:
            cmd = ["./"+str(model.submit_script),str(model.StudyConfig_path)]

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





if __name__ == '__main__':
    unittest.main()
