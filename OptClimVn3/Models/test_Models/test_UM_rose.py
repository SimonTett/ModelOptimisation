import os
import unittest
from unittest.mock import patch, MagicMock

import metomi.rose.config


from UM_rose import UM_rose, UKESM1_1, UKESM1_1_c8
import copy
import tempfile
import pathlib
import genericLib
import shutil
import tarfile



genericLib.setup_env() # set up default env.


class test_umRose(unittest.TestCase):
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

        tmp_dir = tempfile.TemporaryDirectory()
        test_dir = pathlib.Path(tmp_dir.name)  # used throughout.
        ref_dir = pathlib.Path(genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-db898')) # ROSE config
        sim_obs_dir = genericLib.expand('$OPTCLIMTOP/test_in')
        self.tmpDir = tmp_dir  # really a way of keeping in context
        self.testDir = test_dir
        # create a model and store it.
        self.refDir = ref_dir
        filepath =os.environ['OPTCLIMTOP']+'/OptClimVn3/configurations/example_UM_rose/references/u-db898/OptClimVn3/configurations/example_UM_rose/references/u-db898'
        post_process = dict(script='$OPTCLIMTOP/OptClimVn3/scripts/comp_obs.py', output_file='obs.nc')
        self.post_process = post_process
        self.model = UKESM1_1(name='testM', reference=ref_dir,
                            model_dir=test_dir/'model1', config_dir='suite',post_process=post_process,
                            parameters=parameters)


        self.config_path = self.model.config_path
        self.model.model_dir.mkdir()
        shutil.copy(sim_obs_dir / '01_GN' / 'h0101' / 'observables.nc',
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
        dct_comp['configs'] = dct_comp['configs'].to_dict() # convert configs to dict for comparison
        self.assertEqual(dct_comp,dct)



    def test_read_param(self):
        """
        Test that read_value works
        :return:
        """
        shutil.copytree(self.refDir,self.model.config_dir,dirs_exist_ok=True)
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

        load_mdct = model.to_dict()
        my_mdct = self.model.to_dict()
        self.assertEqual(load_mdct,my_mdct)


        ## Check that suite.rc was updated correctly
        with open(model.model_dir/model.config_dir / 'suite.rc', 'r') as suite_file:
            last_line = suite_file.readlines()[-1]
            self.assertEqual(last_line.strip(), "%include optclim.rc")

        ## Check that all the required files were copied over
        config_direct = model.model_dir / model.config_dir
        suite_files = list(config_direct.rglob('*'))
        required_files = [
            'optclim.rc',
            'bin/optclim_task.sh',
        ]
        for filename in required_files:
            self.assertIn(config_direct / filename, suite_files)
        # find all the .bak files in suite_dir
        bak_files = set(config_direct.rglob('*.bak'))
        # expect the following backup files
        expected_bak_files = set([config_direct/(f+'.bak') for f in [
            "app/postproc/rose-app.conf", # We change the postproc script
            "app/um/rose-app.conf", # coz we change variables.
            "rose-suite.conf", # coz we modify variables in here too.
            "suite.rc",# coz we change it by adding an include oprclim.rc
            ]])
        # check have what we expect.
        self.assertEqual(bak_files, expected_bak_files)

        archer_archive_dir = model.read_param('archer_archive_dir')
        self.assertEqual(archer_archive_dir,(model.model_dir/'output').as_posix())




    def test_set_params(self ):
        """
        test set_params method
        :return:
        """
        # copy the ref dir into the suite_dir
        shutil.copytree(self.refDir,self.model.config_dir,dirs_exist_ok=True)
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
        and adding the submit, continue & clean scripts.
        :return:
        """
        shutil.rmtree(self.model.config_dir, onerror=genericLib.errorRemoveReadonly)
        shutil.copytree(self.refDir,self.model.config_dir,dirs_exist_ok=True)
        # create the submit and continue scripts
        for file in [self.model.submit_script, self.model.continue_script,self.model.clean_script]:
            file.parent.mkdir(parents=True, exist_ok=True)
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

        jid=model.running() #
        self.assertEqual(jid,'NOJID')



    def notest_succeeded(self):
        # test that succeeded works. No need as using archive functionality-- code left for now.
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
        expected_prebuild = pathlib.PurePath('~tetts/prebuilds/u-dr496/fcm_make_um').as_posix()  # expected prebuild path
        post_process = dict(script='$OPTCLIMTOP/OptClimVn3/scripts/comp_obs.py', output_file='obs.json')

        parameters = dict(dp_corr_strat=500.0, two_d_fsd_factor=2,
                          ent_fac_dp=1.0, ai=3e-2, RUN_TARGET='P2M')

        run_info = dict(
            prebuild='~tetts/prebuilds/u-dr496/fcm_make_um',  #
            transfer_dir='some_test_dir',# transfer directory
            local_root_dir=self.testDir,
        )
        with patch.multiple(pathlib.Path, is_dir=MagicMock(return_value=True),
                            is_absolute=MagicMock(return_value=True),
                            samefile=MagicMock(return_value=False),):

            model = UM_rose(name='001test', reference=reference,
                            model_dir=self.testDir/'fred/001test', post_process=post_process,
                            parameters=parameters,
                            run_info=run_info)
            self.assertEqual(expected_prebuild,model.parameters_no_key['prebuild'], )
            self.assertEqual('some_test_dir',model.parameters_no_key['transfer_dir'] )
            # check suite_name is as expected.
            self.assertEqual('fred/X001test',model.suite_name, )

    def test_copy(self):
        # test that copy method works
        # Only need to check that have workflow and scripts copied over.
        model = self.model
        model.instantiate()
        cp_dir = self.testDir / 'copy_model'
        model_copy = model.copy(cp_dir)
        # check that the workflow and scripts are copied over.
        for dct in [model_copy.model_dir,model_copy.model_dir/model_copy.script_dir,model_copy.config_dir]:
            self.assertTrue(dct.is_dir(),msg=f'Directory {dct} not copied correctly')


        orig_files = list(model.model_dir.rglob('*'))
        copy_files = list(model_copy.model_dir.rglob('*'))

        # get relative paths for comparison
        orig_rel_paths = sorted([f.relative_to(model.model_dir) for f in orig_files])
        copy_rel_paths = sorted([f.relative_to(model_copy.model_dir) for f in copy_files])
        self.assertEqual(orig_rel_paths, copy_rel_paths, msg=f"Files in {model.model_dir} not copied correctly")

    def no_test_archive(self):
        # Test that archive method works. Rather similar to test_Model.test_archive
        # no longer needed as uses Model version and its own copy
        model = self.model
        model.instantiate()
        model.set_status('SUCCEEDED',check_existing=False)
        archive_file = self.testDir / 'test_archive.tar'
        pp_file = model.model_dir / self.model._post_process_output
        with tarfile.open(archive_file, "w", dereference=True) as archive:
            self.model.archive(archive, self.testDir,
                               extra_files=[self.model._post_process_output])  # archive the model

        # now can try and read it.
        script_dir = model.model_dir / model.script_dir
        expect_paths = ([model.config_path,model.config_dir,script_dir] + list(model.config_dir.rglob('*'))+
                        list(script_dir.rglob('*')))
        expected_names = sorted([p.relative_to(model.model_dir) for p in expect_paths])

        with tarfile.open(archive_file, "r") as archive:
            names = sorted([pathlib.Path(n) for n in archive.getnames()])  # list of names
            self.assertEqual(expected_names, names)

        # now create some obs... and test that works.
        import json
        test_obs = dict(obs1=2.2, obs2=1.0, obs4=True)
        with open(pp_file, 'wt') as fp:
            json.dump(test_obs, fp)
        with tarfile.open(archive_file, "w") as archive:
            model.archive(archive, self.testDir)  # archive the model
        expected_names += [pp_file.relative_to(model.model_dir)]
        expected_names = sorted(expected_names)

        with tarfile.open(archive_file, "r") as archive:
            names = sorted([pathlib.Path(n) for n in archive.getnames()] ) # list of names

            self.assertEqual(expected_names, names)



    def no_test_guess_prebuild(self):
        # check _guess_prebuild works. Turned off aas not using guess_prebuild
        # but left in (for now) in case needed later.
        model = self.model
        model.reference=pathlib.Path('/home/n02/n02-puma/tetts/roses/u-db898')
        expected_path = pathlib.PurePath(f'/home/n02/n02/tetts/cylc-run/u-db898/share/fcm_make_um') # path on puma
        with patch.multiple(pathlib.Path,is_dir=MagicMock(return_value=True),
                            is_absolute=MagicMock(return_value=True)):
            result = model.guess_prebuild(True)
            self.assertEqual(result,expected_path)

        with patch.multiple(pathlib.Path,is_dir=MagicMock(return_value=False),
                            is_absolute=MagicMock(return_value=True)):
            result = model.guess_prebuild(True) # directory guessed does not exist
            self.assertIsNone(result)

        # pass in None and False and should get None
        for val in [None,False]:
            result = model.guess_prebuild(val)
            self.assertIsNone(result)


## specific tests.

# for UKESM1_params want to test that the parameter functions work.
# Will do this by getting hold of default parameters and checking that the functions gives the values in the reference model.
# So idea will be  to have a dict of default values indexed by parameter name. Then loop over that doing the test.
# might be a good place to use subTest.
# Parameters are:
# 'tupp_io',
#  'f0_io',
#  'nl0_io',
#  'rootd_ft_io',
#  'fsmc_p0_io',
#  'gs_nvg_io',
#  'n_lai_exposed',
#  'unload_rate_u',


class TestUKESM1ParamFunctions(unittest.TestCase):
    def setUp(self):

        tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(tmpDir.name)
        refDir1 = pathlib.Path(genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-db898'))
        refDir2 = pathlib.Path(
            genericLib.expand('$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-dr157'))
        post_process = dict(script='$OPTCLIMTOP/OptClimVn3/scripts/comp_obs.py', output_file='obs.json')
        self.tmpDir = tmpDir
        self.testDir = testDir

        testDir1= testDir / 'test1'
        testDir2 = testDir / 'test2'
        self.models = [UKESM1_1(name='test_c7', reference=refDir1,
                              model_dir=testDir1, config_dir='suite', post_process=post_process),
                    UKESM1_1_c8(name='test_c8', reference=refDir2,
                              model_dir=testDir2, config_dir='suite', post_process=post_process)]
        for model in self.models:
            model.instantiate()


    def tearDown(self):
        shutil.rmtree(self.testDir, onerror=genericLib.errorRemoveReadonly)
        self.tmpDir.cleanup()

    def test_UKESM1_param_functions(self):
        """
        Test that UKESM1 parameter functions return the expected default values.
        """
        default_params = {
            'tupp_io': 43., # UKESM1_1 default value
            'f0_io': 0.875,
            'nl0_io': 0.046, # UKESM1_1 default value
            'rootd_ft_io': 2.0,
            'fsmc_p0_io': 0.0, # UKESM1_1 default value
            'gs_nvg_io': 0.01,
            'n_lai_exposed': 27.0, # namelist differently understood in UKESM1_1
            'unload_rate_u': 2.31e-6,
            'cca_md_knob': 0.1, # UKESM1_1 default value
            'aparam': [0.07, 0.0066], # UKESM1_1 default values for aparam and liu_latent. Note default value different from MO value.
            'rho_snow_fresh': [109.0,41.], # UKESM1_1 default value
            'starticetkelvin': [263.15, 0.48717948717948645]  # UKESM1_1 default value

        }
        default_nl_values= {
            'tupp_io': [43,43,43,26,32,32,32,32,45,45,45,40,36],
            'f0_io': [0.875,0.875,0.892,0.875,0.875,0.931,0.931,0.931,0.8,0.8,0.8,0.875,0.875],
            'nl0_io': [0.046,0.046,0.046,0.033,0.033,0.073,0.073,0.073]+5*[0.06],
            'rootd_ft_io': [2,3,2,2,1.8]+8*[0.5],
            'fsmc_p0_io': 13*[0],
            'gs_nvg_io': [0.00000,0.00000,1.00000e-2]+11*[1.00000e+6],
            'n_lai_exposed': [2.0,2.0,27.0,1.0,2.0]+6*[27.0]+[6.0,6.0], # h
            'unload_rate_u': [0.0,0.0,0.0,2.31e-06,2.31e-06]+8*[0.0],
            'cca_md_knob': [0.1,0.1],
            'aparam': [0.07, -0.14 ],  # UKESM1_1 default value
            'rho_snow_fresh': [109.0, 150.],  # UKESM1_1 default value
            'starticetkelvin':[263.15,-20.0]
        }
        # test cylc7 and cylc8 versions
        for model in self.models:
            for param, expected in default_params.items():
                with self.subTest(param=param):
                    print(f'testing {param}')
                    value = model.read_param(param)
                    # check the value is as expected.
                    if isinstance(value, list):
                        self.assertEqual(len(value), len(expected))
                        for idx in range(0,len(value)):
                            self.assertAlmostEqual(value[idx], expected[idx],
                                                   msg=f"{param} at index {idx} did not match reference value")
                    else:
                        self.assertAlmostEqual(value, expected, msg=f"{param} did not match reference value")
                    # read in the values from the namelist.
                    nl_stuff = model.get_param_info(param)[0](model,0.0)  # call the function with a dummy value.
                    nl_values = [model.read_nl_value(nl) for nl,v in nl_stuff]
                    if len(nl_values) == 1:
                        nl_values = nl_values[0]
                    self.assertEqual(nl_values, default_nl_values[param],
                                     msg=f"{param} namelist values did not match reference values {default_nl_values[param]} got {nl_values}")

        # test can read the simple params and that they are the same as the inverse ones.
        expected_values = ['2010-10-01T00:00:00','1979-01-01T00:00:00'] # this is what is exected on the read.
        model_nl_values = [[2010,10,1,0,0,0],[1979,1,1,0,0,0] ] # and these are the namelist values.
        for model,expected_value,nl_value in zip(self.models,expected_values,model_nl_values):
            value = model.read_param('START_TIME')
            self.assertEqual(value, expected_value, msg=f"{type(model)} START_TIME did not match reference value {expected_value} got {value}")
            # read in the values using the function
            value = model.start_time(transform=False) # raw read.
            self.assertEqual(value, nl_value,msg=f"{type(model)} START_TIME namelist values did not match reference values {nl_value} got {value}")
        # check calendar values
        expected_calendars = ['standard', '360_day']  # expected calendars for UKESM1_1 and UKESM1_1_c8
        for model, expected_calendar in zip(self.models, expected_calendars):
            cal = model.calendar()
            self.assertEqual(cal, expected_calendar,
                             msg=f"{type(model)} CALENDAR did not match reference value {expected_calendar} got {cal}")


if __name__ == '__main__':
    unittest.main()
