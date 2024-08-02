# test the scripts
import pathlib
import unittest
import subprocess
import tempfile
from Model import Model
import platform
import shutil

import engine
import genericLib
import os

import StudyConfig
from runSubmit import runSubmit # so we can test if we have one!


class testScripts(unittest.TestCase):

    def setup_model(self):
        cpath = Model.expand("$OPTCLIMTOP/OptClimVn3/configurations/example_simple_model")
        eng = engine.abstractEngine.create_engine('SGE')

        model = Model('test_model',
                            reference=cpath,engine=eng,
                            config_path=self.tempDir / 'testmodel.mcfg',
                            model_dir=self.tempDir / 'testmodel',
                            parameters=dict(pone=2, pthree=3),
                            status='SUBMITTED')
        model.model_dir.mkdir(exist_ok=True, parents=True)
        model.dump_model()
        return model

    def setUp(self) -> None:
        direct = tempfile.TemporaryDirectory()
        self.direct = direct
        self.tempDir = pathlib.Path(direct.name)
        self.script_dir = Model.expand("$OPTCLIMTOP/OptClimVn3/scripts")
        self.assertTrue(self.script_dir.exists())

    def tearDown(self) -> None:
        """
        Clean up by removing the temp directory contents
        :return:
        """
        shutil.rmtree(self.direct.name, onerror=genericLib.errorRemoveReadonly)
        self.direct.cleanup()


    def test_set_model_status(self):
        # test set_model_status works
        # need  a model with status = "SUBMITTED" setup_model does that.

        os.environ['JOB_ID']='123456'
        os.environ['SLURM_JOB_ID']='123456'
        pth = self.script_dir/"set_model_status.py"
        self.assertTrue(pth.exists())
        model = self.setup_model()

        if platform.system() == 'Windows':
            cmd = ['python',str(pth)]
        else:
            cmd = [str(pth)]
        cmd += [str(model.config_path),'RUNNING','-v']
        try:
            subprocess.run(cmd,cwd=model.model_dir,capture_output=True,check=True,text=True)
        except subprocess.CalledProcessError as err:
            print("stdout",err.stdout)
            print("stderr",err.stderr)
            raise
        model = Model.load_model(model.config_path)
        self.assertEqual(model.status,'RUNNING')
        model.delete()



    def test_runAlgorithm(self):
        """ Test runAlgorithm by running it in test mode.
        Success means it has worked!
        """
        if platform.system() == 'Windows':
            cmd=['python']

        else:
            cmd=[]
        config_pth = Model.expand("$OPTCLIMTOP/OptClimVn3/configurations/dfols14param_opt3.json")
        config = StudyConfig.readConfig(config_pth)
        cmd += [str(self.script_dir/'runAlgorithm.py'),str(config_pth),"-v", "-t",
                "-d",str(self.tempDir) ,"--delete"]
        res = subprocess.run(cmd,capture_output=True,text=True)
        if res.returncode != 0:
            print("stdout",res.stdout)
            print("stderr",res.stderr)
            res.check_returncode()
        config_pth = self.tempDir/(config.name() + ".scfg")
        self.assertTrue(config_pth.exists()) # check config file exists.
        sconfig = runSubmit.load(config_pth)
        self.assertIsInstance(sconfig,runSubmit)



if __name__ == '__main__':
    unittest.main()
