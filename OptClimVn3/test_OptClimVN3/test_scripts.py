# test the scripts
import pathlib
import sys
import unittest
import subprocess
import tempfile
from Model import Model
import platform
import shutil
import typing

import genericLib
import os
import engine

import StudyConfig
from runSubmit import runSubmit  # so we can test if we have one!

genericLib.setup_env()


class testScripts(unittest.TestCase):

    def setup_model(self):
        cpath = Model.expand("$OPTCLIMTOP/OptClimVn3/configurations/example_simple_model")
        eng = engine.abstractEngine.guess_engine()

        model = Model('test_model',
                      reference=cpath, engine=eng,
                      config_path=self.tempDir / 'testmodel.mcfg',
                      model_dir=self.tempDir / 'testmodel',fake=True,
                      parameters=dict(pone=2, pthree=3), status='SUBMITTED')
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
        # need a model with status = "SUBMITTED" setup_model does that.

        os.environ['JOB_ID'] = '123456'
        os.environ['SLURM_JOB_ID'] = '123456'
        pth = self.script_dir / "set_model_status.py"
        self.assertTrue(pth.exists())
        model = self.setup_model()

        if platform.system() == 'Windows':
            cmd = [sys.executable, str(pth)]
        else:
            cmd = [str(pth)]
        cmd += [str(model.config_path), 'RUNNING', '-v']
        try:
            subprocess.run(cmd, cwd=model.model_dir, capture_output=True, check=True, text=True)
        except subprocess.CalledProcessError as err:
            print("stdout", err.stdout)
            print("stderr", err.stderr)
            raise
        model = Model.load_model(model.config_path)
        self.assertEqual(model.status, 'RUNNING')
        model.delete()

    def test_runAlgorithm(self):
        """ Test runAlgorithm by running it in test mode.
        Success means it has worked!
        """
        def make_cmd(config_path:typing.Optional[pathlib.Path]=None):
            if platform.system() == 'Windows':
                cmd = [sys.executable]

            else:
                cmd = []
            if config_path is None:
                config_path = Model.expand("$OPTCLIMTOP/OptClimVn3/configurations/dfols14param_opt3.json")
            cmd += [str(self.script_dir / 'runAlgorithm.py'), str(config_path), "-v", "-t",
                    "-d", str(self.tempDir)]
            return cmd

        config_path = Model.expand("$OPTCLIMTOP/OptClimVn3/configurations/dfols14param_opt3.json")
        cmd = make_cmd(config_path=config_path)
        config = StudyConfig.readConfig(config_path)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("stdout", res.stdout)
            print("stderr", res.stderr)
            res.check_returncode()
        config_pth = self.tempDir / (config.name() + ".scfg")
        self.assertTrue(config_pth.exists())  # check config file exists.
        sconfig = runSubmit.load(config_pth)
        self.assertIsInstance(sconfig, runSubmit)
        # check that config.run_info()['local_root_dir'] is self.tempDir.parent
        local_root_dir = sconfig.config.run_info()['local_root_dir']
        self.assertEqual(self.tempDir, pathlib.Path(local_root_dir))


        cmd +=['--no-set_local_root_dir']
        shutil.rmtree(self.tempDir)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("stdout", res.stdout)
            print("stderr", res.stderr)
            res.check_returncode()
        config_pth = self.tempDir / (config.name() + ".scfg")
        sconfig = runSubmit.load(config_pth)
        self.assertIsNone(sconfig.config.run_info().get('local_root_dir'))
        # now use dryrun
        cmd = make_cmd()
        cmd += ['--dryrun','-v']
        shutil.rmtree(self.tempDir)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("stdout", res.stdout)
            print("stderr", res.stderr)
            print("cmd is ", ' '.join(cmd))
            res.check_returncode()

        config_pth = self.tempDir / (config.name() + ".scfg")
        sconfig = runSubmit.load(config_pth)
        print(len(sconfig.model_index))
        models = [m for m in sconfig.model_index.values() if m.status == 'INSTANTIATED']
        # fake run the first 5 models.
        for m in models[:5]:
            m.status = 'PROCESSED'
            m.simulated_obs = genericLib.fake_fn(sconfig.config,m.parameters)
            m.dump_model()


        res = subprocess.run(cmd, capture_output=True, text=True)
        print("stdout", res.stdout)
        print("stderr", res.stderr)
        if res.returncode != 0:

            print("cmd is ", ' '.join(cmd))
            res.check_returncode()
        sconfig = runSubmit.load(config_pth)
        models = sconfig.processed_models()
        self.assertEqual(len(models),5)
        self.assertEqual(len(sconfig.logical_obs()),5)
        self.assertEqual(len(sconfig.logical_cost()),5)


if __name__ == '__main__':
    unittest.main()
