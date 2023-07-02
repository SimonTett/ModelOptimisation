# test the scripts
import pathlib
import unittest
import subprocess
import tempfile
import Model
import platform

import StudyConfig
from runSubmit import runSubmit # so we can test if we have one!


class testScripts(unittest.TestCase):

    def setup_model(self):
        cpath = Model.Model.expand("$OPTCLIMTOP/OptClimVn3/configurations/example_simple_model")
        model = Model.Model('test_model',
                            reference=cpath,
                            config_path=self.tempDir / 'testmodel.mcfg',
                            model_dir=self.tempDir / 'testmodel',
                            parameters=dict(pone=2, pthree=3),
                            status='SUBMITTED')
        model.model_dir.mkdir(exist_ok=True, parents=True)
        model.dump_model()
        model.setup_model_env()  # setup the env.
        return model

    def setUp(self) -> None:
        direct = tempfile.TemporaryDirectory()
        self.direct = direct
        self.tempDir = pathlib.Path(direct.name)
        self.script_dir = Model.Model.expand("$OPTCLIMTOP/OptClimVn3/scripts")
        self.assertTrue(self.script_dir.exists()) # this failing when ran along with all tests.

    def test_set_model_status(self):
        # test set_model_status works
        # need  a model with status = "SUBMITTED" setup_model does that.
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
        model = Model.Model.load_model(model.config_path)
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
        config_pth = Model.Model.expand("$OPTCLIMTOP/OptClimVn3/configurations/dfols14param_opt3.json")
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
