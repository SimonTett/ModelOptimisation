import os
import unittest
from unittest.mock import patch, MagicMock

import metomi.rose.config
from aiofiles.ospath import samefile
from scipy.constants import value

from UM_rose import UM_rose
import copy
import tempfile
import pathlib
import genericLib
import shutil



genericLib.setup_env() # set up default env.


class test_um_rose(unittest.TestCase):
    def setUp(self):
        """
        Setup case
        :return:
        """
        parameters = dict(
            dp_corr_strat=500.0,two_d_fsd_factor=2,
            ent_fac_dp= 1.0, ai=3e-2
                          )
        self.parameters = copy.deepcopy(parameters)

        tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(tmpDir.name)  # used throughout.
        refDir = pathlib.Path(genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-db898')) # ROSE config
        simObsDir = genericLib.expand('$OPTCLIMTOP/test_in')
        self.tmpDir = tmpDir  # really a way of keeping in context
        self.testDir = testDir
        # create a model and store it.
        self.refDir = refDir
        filepath =os.environ['OPTCLIMTOP']+'/OptClimVn3/configurations/example_UM_rose/references/u-db898/OptClimVn3/configurations/example_UM_rose/references/u-db898'
        post_process = dict(script='$OPTCLIMTOP/OptClimVn3/scripts/comp_obs.py', output_file='obs.json')
        self.post_process = post_process
        self.model = UM_rose(name='testM', reference=refDir,
                            model_dir=testDir, suite_dir=testDir/'suite',post_process=post_process,
                            parameters=parameters)

        self.config_path = self.model.config_path

        shutil.copy(simObsDir / '01_GN' / 'h0101' / 'observables.nc',
                    self.model.model_dir / 'obs.nc')  # copy over a netcdf file of observations.

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        shutil.rmtree(self.testDir, onerror=genericLib.errorRemoveReadonly)
        self.tmpDir.cleanup()

    def test_to_dict(self ):
        """
        Test to_dict method works.
        Test that _config_cache not in the directory but everything else the same
        :return:
        """

        dct = self.model.to_dict()
        dct_comp = vars(self.model)
        dct_comp.pop('configs')
        self.assertEqual(dct_comp,dct)



    def test_read_param(self):
        """
        Test that read_value works
        :return:
        """
        shutil.copytree(self.refDir,self.model.suite_dir,dirs_exist_ok=True)
        val=self.model.read_param('ai')
        self.assertEqual(val,2.5700e-02)

    def test_instantiate(self):
        """
        Test instantiation.
        :return:
        """
        # possible tests
        # 1) Test that changed name and directory has been changed.
        #  Note name and directory don't need to be the same.
        """
        Test instantiate works!
        :return:
        """

        self.model.instantiate()
        p = set(self.parameters.keys())

        params_got = self.model.read_params(list(p))
        self.assertEqual(params_got,self.parameters)


        self.assertEqual(self.model.status,'INSTANTIATED')
        model = self.model.load_model(self.model.config_path)


        self.assertEqual(model.to_dict(),self.model.to_dict())

        ## Check that suite.rc was updated correctly
        with open(model.suite_dir / 'suite.rc', 'r') as suite_file:
            last_line = suite_file.readlines()[-1]
            self.assertEqual(last_line.strip(), "%include optclim.rc")

        ## Check that all the required files were copied over
        suite_files = list(model.model_dir.rglob('*'))
        required_files = [
            'optclim.rc',
            'bin/optclim_task.sh',
        ]
        for filename in required_files:
            self.assertIn(model.suite_dir / filename, suite_files)
        # find all the .bak files in suite_dir
        bak_files = set(model.suite_dir.rglob('*.bak'))
        # expect the following backup files
        expected_bak_files = set([model.suite_dir/(f+'.bak') for f in [
            "app/um/rose-app.conf", # coz we change variables.
            "rose-suite.conf", # coz we modify variables in here too.
            "suite.rc",# coz we change it by adding an include oprclim.rc
            ]])
        # check have what we expect.
        self.assertEqual(bak_files, expected_bak_files)
        # check that tar file generated
        tar_file = self.model.model_dir / 'rose_suite.tar.gz'
        self.assertTrue(tar_file.exists())
        # check that the tar file has at least 1K byte in it
        self.assertGreater(tar_file.stat().st_size, 1000)



    def test_set_params(self ):
        """
        test set_params method
        :return:
        """
        # copy the ref dir into the suite_dir
        shutil.copytree(self.refDir,self.model.suite_dir,dirs_exist_ok=True)
        params = dict(ai=1e-2,dp_corr_strat=500.0,two_d_fsd_factor=2,ent_fac_dp= 1.0)
        self.model.set_params(parameters=params)
        # test values
        for p,v in params.items():
            self.assertEqual(v,self.model.read_param(p))


    def test_run_time(self):
        """
        Test run time works
        :return:
        """

        # test that run time works
        from namelist_var import NamelistVar
        expect_nl = NamelistVar('um_rose', filepath=pathlib.Path('rose-suite.conf'),
                                namelist='jinja2:suite.rc', nl_var='MAIN_CLOCK', default=0)
        for time,time_str in zip(
            [60*60.,'PT16M40S'],
            ['PT1H','PT16M40S']
        ):
            expected = [(expect_nl,time_str)]
            got = self.model.run_time(time)
            self.assertEqual(expected,got,msg=f'Expected {expected} got {got}')
        self.model.instantiate()
        got = self.model.run_time(None)
        self.assertEqual(got,self.model.read_nl_value(expect_nl))

    def test_replace_file(self):
        # Test replace_file works
        file = self.model.model_dir / 'stuff.text'
        with open(file, 'w') as f:
            f.write("""
            This is a test file.
            It should be replaced.
            [file:$ROOT_DIR/rose-suite.conf]
            some more text
            and even more text
            """)
        result = self.model.replace_file(file,match='^NO MATCH$',replacement='fred')
        # nothing should have changed so should be None
        self.assertIsNone(result)
        # now test a match
        new_file,matches = self.model.replace_file(file,match='^.*rose-suite.conf.*$',
                                                   replacement='fred',backup_ext='.bak')
        self.assertEqual(new_file, file)
        self.assertEqual(matches,1)
        # should have a backup file
        backup=genericLib.backup_file(file,ext='.bak')
        self.assertTrue(backup.exists())


    def notest_change_rose_dir(self):
        # Test change_rose_dir works. No longer required. Code left for now.
        testdir = self.model.model_dir / 'testing'
        testdir.mkdir(parents=True, exist_ok=True)
        file1 = testdir / 'stuff.text'
        with file1.open('w') as f:
            f.write("""
This is a test file.
It should be replaced.
[file:$ROSE_DATA/rose-suite.conf]
some more text
and even more text
  [file:$ROSE_DATA/rose-suite2.conf]
   a bit more text
[file:$ROSE_DATA/etc/ancil/qrclim.biog]
some values
                    """)
        file2 = testdir / 'stuff2.text'
        with file2.open('w') as f:
            f.write("""
This is a test file.
[!!file:$ROSE_DATA/rose-suite.conf]
some more text
and even more text
                    """)
        changed_files = self.model.change_rose_dir(testdir)
        self.assertEqual({file1:3},changed_files)

    def reinit(self):
        """
        Utility method to Reinitialise the model by copying the reference directory to the suite_dir
        and adding the submit and continue scripts.
        :return:
        """
        shutil.rmtree(self.model.suite_dir, onerror=genericLib.errorRemoveReadonly)
        shutil.copytree(self.refDir,self.model.suite_dir,dirs_exist_ok=True)
        # create the submit and continue scripts
        for file in [self.model.submit_script, self.model.continue_script]:

            with file.open('wt') as f:
                f.write('A script')

    def test_check(self):
        """
        Test check method
        :return:
        """


        self.reinit()


        # test that check works.
        ok=self.model.check()
        self.assertTrue(ok)
        # now modify parameters so that check should fail.
        self.model.set_params(dict(RESUB_TIME='P1M1D'))
        with self.assertRaises(ValueError) as cm:
            self.model.check()
        self.reinit()
        self.model.set_params(dict(RESUB_TIME='P1M',RUN_TARGET='P1Y3M1D'))
        with self.assertRaises(ValueError) as cm:
            self.model.check()
        self.reinit()
        self.model.set_params(dict(RUN_TARGET='P1TY3M1D')) # malformed time
        with self.assertRaises(ValueError) as cm:
            self.model.check()

        self.reinit()
        self.model.set_params(dict(RUN_TARGET='P1Y3M'))  # Ok time!
        self.model.submit_script.unlink() # remove the submit script
        with self.assertRaises(FileNotFoundError) as cm:
            self.model.check()





    def test_running(self):
        # test for UM_rose running version.
        # Does the following tests:
        # 1) Test that model_data_dir is set to the correct directory based on the environment variable ROSE_DATA and DATAM
        # 2) Test that model_data_dir is unmodified if not None
        # 3) Test that model_data_dir is not changed if ROSE_DATA is not in the environment

        # set up environment
        model = self.model
        model.instantiate()
        model.set_status('SUBMITTED')
        os.environ['ROSE_DATA'] = str(self.model.model_dir/'data')
        os.environ['DATAM']='History_Data'
        model_data_dir=pathlib.Path(os.environ['ROSE_DATA'])/os.environ['DATAM']
        model_data_dir.mkdir(exist_ok=True,parents=True) # create the directory if it doesn't exist.
        model.running() #
        self.assertEqual(model_data_dir,model.model_data_dir)
        # set model.model_data_dir to model_dir. Should not be modified,
        model.model_data_dir = model.model_dir
        model.set_status('SUBMITTED',check_existing=False)
        model.running()
        # check that model_data_dir is set to model_dir
        self.assertEqual(self.model.model_data_dir,self.model.model_dir)
        # remove ROSE_DATA from the environment and set self.model.model_data_dir to None
        del os.environ['ROSE_DATA']
        model.model_data_dir = None
        # test that running works.
        model.set_status('SUBMITTED',check_existing=False)
        model.running()
        self.assertIsNone(self.model.model_data_dir) # model_data_dir should be None

    def test_succeeded(self):
        # test that succeeded works.
        # Will check that files are successfully copied from model_data_dir to model_dir/History_Data
        model = self.model
        model.model_data_dir = self.model.model_dir/'test_data'
        model.model_data_dir.mkdir(exist_ok=True,parents=True) # create the directory if it doesn't exist.
        model.set_status('RUNNING',check_existing=False)
        files = ['file1.pp', 'file2.pp','file1.nc','file2.nc']
        dumps = ['file.d1_00','file.d2_00','file.d3_00']
        expected = files + dumps[-1:]
        # create the files in the model_data_dir
        for file in files+dumps:
            with open(model.model_data_dir / file, 'wt') as f:
                f.write(f'test file {file}')

        model.succeeded()
        # check that the files are copied to model_dir/'share/data/History_Data'
        got_files = list((model.model_dir/'share/data/History_Data').glob('*'))
        got_files = set([f.name for f in got_files])
        self.assertEqual(got_files, set(expected))

    def test_init(self):
        # test the init method
        reference = pathlib.Path('/home/n02/n02-puma/tetts/roses/u-db898')
        expected_prebuild = pathlib.PurePath(f'/home/n02/n02/tetts/cylc-run/u-db898/share/fcm_make_um')
        post_process = dict(script='$OPTCLIMTOP/OptClimVn3/scripts/comp_obs.py', output_file='obs.json')

        parameters = dict(dp_corr_strat=500.0, two_d_fsd_factor=2,
                          ent_fac_dp=1.0, ai=3e-2, RUN_TARGET='P2M')

        run_info = dict(
            prebuild=True,  # guess the prebuild file
            use_scratch=True  # use scratch space. Means models get cleaned up after 28 days.
        )
        with patch.multiple(pathlib.Path, is_dir=MagicMock(return_value=True),
                            is_absolute=MagicMock(return_value=True),
                            samefile=MagicMock(return_value=False),):

            model = UM_rose(name='fred', reference=reference,
                            model_dir=self.testDir, post_process=post_process,
                            parameters=parameters,
                            run_info=run_info)
            self.assertEqual(model.parameters_no_key['prebuild'], str(expected_prebuild))
    def test__guess_prebuild(self):
        # check __guess_prebuild works
        model = self.model
        model.reference=pathlib.Path('/home/n02/n02-puma/tetts/roses/u-db898')
        expected_path = pathlib.Path(f'/home/n02/n02/tetts/cylc-run/u-db898/share/fcm_make_um') # path on puma
        with patch.multiple(pathlib.Path,is_dir=MagicMock(return_value=True),
                            is_absolute=MagicMock(return_value=True)):
            result = model._guess_prebuild()
            self.assertEqual(result,expected_path)

        with patch.multiple(pathlib.Path,is_dir=MagicMock(return_value=False),
                            is_absolute=MagicMock(return_value=True)):
            result = model._guess_prebuild()
            self.assertIsNone(result)







        



if __name__ == '__main__':
    unittest.main()
