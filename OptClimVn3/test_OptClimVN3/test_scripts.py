# test the scripts
import copy
import pathlib
import sys
import time
import unittest
import subprocess
import tempfile
from Model import Model
import platform
import shutil
import contextlib
import typing

import genericLib
import os
import engine

import StudyConfig
from runSubmit import runSubmit  # so we can test if we have one!
import archive_study
genericLib.setup_env()

import warnings

# internal flag so we only warn once per process if the stdlib provides chdir
_CONTEXT_CHDIR_WARNED = False


@contextlib.contextmanager
def context_chdir(path: typing.Union[str, pathlib.Path]):
    """Context manager that temporarily changes CWD and restores it on exit.

    If the running Python's stdlib already provides `contextlib.chdir`, this helper
    will delegate to it but emit a single, first-time-only warning suggesting the
    stdlib alternative. On older Pythons (e.g. 3.10) it falls back to a simple
    os.chdir try/finally implementation. The context yields a pathlib.Path for
    compatibility with code that used ``with contextlib.chdir(...) as newdir:``.
    """
    global _CONTEXT_CHDIR_WARNED

    # Prefer the stdlib implementation when available (Python 3.11+), but
    # issue a one-time informational warning so callers know they can switch.
    if hasattr(contextlib, 'chdir'):
        if not _CONTEXT_CHDIR_WARNED:
            warnings.warn(
                "stdlib contextlib.chdir is available in this Python; consider using it instead of the local fallback.",
                UserWarning,
                stacklevel=2,
            )
            _CONTEXT_CHDIR_WARNED = True
        # Delegate to the stdlib context manager for correct behaviour
        with contextlib.chdir(path) as p:
            yield pathlib.Path(p) if p is not None else pathlib.Path(str(path))
        return

    # Fallback for older Pythons
    prev = os.getcwd()
    target = str(path)
    os.chdir(target)
    try:
        yield pathlib.Path(target)
    finally:
        os.chdir(prev)

def run_cmd(*args):
    """
    Run a python command is sys indep way. On windows shove sys.executable in front of args
    Print out stdout and stderr if command failed.
    :param args:
    :return:
    """
    if platform.system() == 'Windows':
        cmd = [sys.executable]

    else:
        cmd = []
    cmd += args

    res = subprocess.run(cmd, capture_output=True, text=True)
    print("stdout", res.stdout)
    print("stderr", res.stderr)
    if res.returncode != 0:

        res.check_returncode()
    return res
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
        self.script_dir = genericLib.expand("$OPTCLIMTOP/OptClimVn3/scripts")
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
        cmd += ['--dryrun']
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
        self.assertEqual(len(sconfig.simulated_observations()),5)
        self.assertEqual(len(sconfig.logical_cost()),5)


    def test_OptClim_control(self):
        # test OptClim_Control.
        # Tests each option and that config after running command is as expected.
        # extract from archive an existing config

        arc, cfg = archive_study.archive_study.extract_archive(
            genericLib.expand('$OPTCLIMTOP/OptClimVn3/test_data/archive_dfols4p.tar.gz'),
            direct=self.tempDir)
        script_path = self.script_dir/'OptClim_control.py'
        cfg_path = self.tempDir/'dfols4p.scfg'
        # update first. For this we need to write out a config file
        config = cfg.config
        config.save(self.tempDir / 'dfols4p.json') # save the json config.
        cfg.dump_config(dump_models=False) # and dump the config (which should update the file but nothing else.)

        config = copy.deepcopy(config) # copy the json configuration.
        config.setv("song_type_comment",'this is not a love song')
        config.save() # save the config which should be changed
        self.assertIsNone(cfg.config.getv('song_type_comment'))
        # First check we can write to a different output file with everythigng working
        new_cfg_path = self.tempDir / 'dfols4p_new/dfols4p_new.scfg'
        run_cmd(str(script_path),str(cfg_path),'update',str(config.fileName()),'--output',str(new_cfg_path))
        cfg2 = runSubmit.load(new_cfg_path) # load it.
        self.assertEqual(cfg2.config.getv('song_type_comment'),'this is not a love song')

        run_cmd(str(script_path),str(cfg_path),'update')
        cfg2 = runSubmit.load(cfg_path) # load it.
        self.assertEqual(cfg2.config.getv('song_type_comment'),'this is not a love song')
        # test reading in from a non default config.
        config2 = copy.deepcopy(config)
        config2.setv('song_type_comment','this is a love song')
        config2_path = self.tempDir / 'dfols4p_final2.json'
        config2.save(config2_path) # save the json config.
        run_cmd(str(script_path), str(cfg_path), 'update',str(config2_path))
        cfg2 = runSubmit.load(cfg_path)  # load it.
        self.assertEqual(cfg2.config.getv('song_type_comment'), 'this is a love song')

        # dump the original config back
        cfg.config.save()
        cfg.dump_config(dump_models=False)  # and dump the config (which should update the file)

        self.assertIsNone(cfg.config.getv('song_type_comment'))
        # now do update where we are in the directory and do not provide a config
        with context_chdir(self.tempDir) as newdir: # for python 3.12 update this to contextlib.chdir
            run_cmd(str(script_path),'update')
            cfg3 = runSubmit.load(cfg.config_path) # load it.
            self.assertIsNone(cfg3.config.getv('song_type_comment'))



        # now for plot
        monitor_file = self.tempDir / 'monitor.png'
        run_cmd(str(script_path), str(cfg_path), 'plot',str(monitor_file))
        self.assertTrue(monitor_file.exists())
        monitor_file.unlink() # remove it
        monitor_file = self.tempDir / f"monitor_{cfg.name}.png"
        with context_chdir(self.tempDir) as newdir:
            run_cmd(str(script_path),str(cfg_path),'plot')
            self.assertTrue(monitor_file.exists())
            monitor_file.unlink() # remove it

        # test stop
        run_cmd(str(script_path),str(cfg_path),'stop')
        cfg = runSubmit.load(cfg_path)
        self.assertEqual(cfg.next_command,'stop')

        # test continue
        run_cmd(str(script_path),str(cfg_path),'continue')
        cfg = runSubmit.load(cfg_path)
        self.assertIsNone(cfg.next_command)

        # test kill...
        run_cmd(str(script_path),str(cfg_path),'kill')
        cfg = runSubmit.load(cfg_path)
        self.assertEqual(list(cfg._history.values())[-1],['Killed 0 jobs']) # should report no jobs killed.




if __name__ == '__main__':
    unittest.main()