import logging
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch, mock_open

import f90nml
import metomi.rose.config

import genericLib
from namelist_var import BaseConfig, NamelistVar, \
    JSON_Config, GroupConfig,FortranNamelistConfig,UMroseNamelistConfig

genericLib.setup_env()
logging.basicConfig(level=logging.INFO,force=True)



import json


def mock_read(allow_missing=False):
    """
    Mock read function
    :param allow_missing:
    :return:
    """
    config = {
        "system":
            {
                "sleep_time": 15,
                "sleep_time_comment": "How many seconds to sleep for",
                "fail_probability": 0.0,
                "fail_probability_comment": "probability of failure"
            },
        "system_comment": "System variables",
        "model_params":
            {
                "CT": 0.0001,
                "EACF": 0.5,
                "ENTCOEF": 3.0,
                "ICE_SIZE": 3e-05,
                "RHCRIT": 0.7,
                "VF1": 1.0,
                "CW": 0.0002,
                "CW_comment": "Seed parameter which  affects CW_SEA & CW_LAND",
                "DYNDIFF": 12.0,
                "DYNDIFF_comment": "Seed parameter which affects DIFF_COEFF, DIFF_COEFF_Q, DIFF_EXP & DIFF_EXP_Q",
                "KAY_GWAVE": 20000.0,
                "KAY_GWAVE_comment": "Seed parameter which also affects KAY_LEE_GWAVE",
                "ASYM_LAMBDA": 0.15,
                "CHARNOCK": 0.012,
                "CHARNOCK_comment": "Note this is Murphy et al, 200X and is different from that reported in Yamazaki et al, 2013",
                "G0": 10.0,
                "Z0FSEA": 0.0013,
                "ALPHAM": 0.5,
                "ALPHAM_comment": "Seed parameter which affects DTICE and ALPHAM"
            },
        "model_params_comment": "parameters for the model. Simple model does not actually care! These come from hadCm3",
        "comment": "Reference params for simple_model_pars_json"
    }

    return config


class test_BaseConfig(unittest.TestCase):

    @patch.object(BaseConfig, 'read', side_effect=mock_read)
    def setUp(self, mck):
        # Setup code here
        # Make  a temp dir and copy the reference case to it.
        self.tmpDir = tempfile.TemporaryDirectory()
        self.refDir = genericLib.expand(
            '$OPTCLIMTOP/OptClimVn3/configurations/example_simple_model_pars_json/reference')
        self.root_dir = pathlib.Path(self.tmpDir.name)
        shutil.copytree(self.refDir, self.root_dir, dirs_exist_ok=True)
        self.rel_filepath = pathlib.Path('parameters.json')
        self.config = BaseConfig(root_dir=self.root_dir, rel_filepath=self.rel_filepath)
        #

    def tearDown(self):
        # Teardown code here
        self.tmpDir.cleanup()

    def test_filepath(self):
        # Test case for filepath method
        # Suggested test case: Check if the returned path is correct
        expected_path = self.root_dir / self.rel_filepath
        self.assertEqual(self.config.filepath(), expected_path)

    def test_check_right_nl(self):
        # Test case for check_right_nl method
        # Check if the method raises ValueError for mismatched type_name
        namelist = NamelistVar(type_name='wrong_type', filepath=self.rel_filepath, namelist='test', nl_var='var')
        with self.assertRaises(ValueError):
            self.config.check_right_nl(namelist)
        # Check if the method raises ValueError for mismatched filename.
        namelist = NamelistVar(type_name='correct_type', filepath=pathlib.Path('wrong_file.json'), namelist='test',
                               nl_var='var')
        with self.assertRaises(ValueError):
            self.config.check_right_nl(namelist)
        #and get True for correct
        namelist = NamelistVar(type_name='base_nl', filepath=self.rel_filepath, namelist='test', nl_var='var')
        self.assertTrue(self.config.check_right_nl(namelist))

    def test_check_ok(self):
        # Test case for check_ok method
        # Suggested test case: Check if the method returns True for valid namelist
        namelist = NamelistVar(type_name='base_nl', filepath=self.rel_filepath, namelist='test', nl_var='var')
        self.config.modified_values[namelist] = False
        self.assertTrue(self.config.check_ok(namelist))
        # and Fails for modified
        self.config.modified_values[namelist] = True
        with self.assertRaises(ValueError):
            self.config.check_ok(namelist)

    def test_backup(self):
        # Test case for backup method
        # Check if the backup file is created correctly if config exists.
        config = self.config
        path = config.backup()
        self.assertFalse(path.exists())

    def test_namelist_names(self):
        # Test case for namelist_names method
        # Check if the method returns the correct list of namelist names
        self.config.config = {'namelist1': {}, 'namelist2': {}}
        self.assertEqual(self.config.namelist_names(), ['namelist1', 'namelist2'])

    def test_var_names(self):
        # Test case for var_names method
        #  Check if the method returns the correct list of variable names for a given namelist
        self.config.config = {'namelist1': {'var1': 1, 'var2': 2}}
        self.assertEqual(self.config.var_names('namelist1'), ['var1', 'var2'])


class TestJSON_Config(unittest.TestCase):

    def setUp(self):
        self.root_dir = genericLib.expand(
            "$OPTCLIMTOP/OptClimVn3/configurations/example_simple_model_pars_json/reference")
        self.rel_filepath = pathlib.Path('parameters.json')
        self.config = JSON_Config(root_dir=self.root_dir, rel_filepath=self.rel_filepath)

    def tearDown(self):
        pass

    @patch('pathlib.Path.open', new_callable=mock_open, read_data='{"key": "value"}')
    def test_read(self, mock_file):
        # Test the read method to ensure it correctly reads JSON data from a file
        config = self.config.read()
        self.assertEqual(config, {"key": "value"})

    @patch('pathlib.Path.open', new_callable=mock_open)  # critical this is here else the config would be overwritten
    @patch('pathlib.Path.rename')  # critical this is here else the config would be overwritten
    def test_write(self, mock_rename, mock_file):
        # Test the write method to ensure it correctly writes JSON data to a file
        self.config.config = {"key": "value"}
        self.config.write()
        mock_file.assert_called_once_with('w+t')
        # Check what would have been written out is as expected
        # Get the file handle used by the mock
        handle = mock_file()
        # Get the actual written data
        written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
        # Expected JSON string
        expected_data = json.dumps({"key": "value"}, indent=2)

        self.assertEqual(written_data, expected_data)
        # check backup happened.
        back_file = self.config.filepath()
        back_file = back_file.parent / (back_file.name + '.bak')
        mock_rename.assert_called_once_with(back_file)

    def test_read_value(self):
        # Test the read_value method to ensure it correctly reads a value from the config
        self.config.config = {"namelist": {"var": 1}}
        namelist = NamelistVar(type_name='json_nl', filepath=self.rel_filepath, namelist='namelist', nl_var='var')
        value = self.config.read_value(namelist)
        self.assertEqual(value, 1)

    def test_update_value(self):
        # Test the update_value method to ensure it correctly updates a value in the config
        self.config.config = {"namelist": {"var": 1}}
        namelist = NamelistVar(type_name='json_nl', filepath=self.rel_filepath, namelist='namelist', nl_var='var')
        self.config.update_value(namelist, 2)
        self.assertEqual(self.config.config["namelist"]["var"], 2)
        self.assertTrue(self.config.modified_values[namelist])
        # check get an error if update twice.
        with self.assertRaises(ValueError):
            self.config.update_value(namelist, 3)

class TestFortranNamelist_Config(unittest.TestCase):

    def setUp(self):
        self.root_dir = genericLib.expand(
            "$OPTCLIMTOP/OptClimVn3/configurations/example_HadAM3/reference")
        self.rel_filepath = pathlib.Path('CNTLATM')
        self.config = FortranNamelistConfig(root_dir=self.root_dir, rel_filepath=self.rel_filepath)

    def tearDown(self):
        pass


    def test_read(self):
        # Test the read method to ensure it correctly reads namelist data from a file
        config = self.config.read()
        # check two distinct namelists/vars
        test={
            ('RUNCNST','ALPHAM'):0.5,
            ('RUNCNST','DIFF_COEFF'):[5.470e+08,5.470e+08,5.470e+08,5.470e+08,
 5.470e+08,5.470e+08,5.470e+08,5.470e+08,5.470e+08,5.470e+08,5.470e+08,
 5.470e+08,5.470e+08,5.470e+08,5.470e+08,5.470e+08,5.470e+08,5.470e+08,
 4.000e+06],
            ('NLSTCATM','L_CLD_area'):False,
            ('NLSTCATM','H_LWBANDS'):8,
        }
        for k,v in test.items():
            self.assertEqual(config[k[0]][k[1]],v)

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.rename')  # critical this is here else the config would be overwritten
    def test_write(self, mock_rename, mock_open):
        # Test the write method to ensure it correctly writes fortran data to a file
        self.config.config = f90nml.Namelist()
        self.config.config['RUNCNST'] = {'ALPHAM': 0.5}
        self.config.write()
        mock_open.assert_called_once_with(self.config.filepath() ,'w')
        # Check what would have been written out is as expected
        # Get the file handle used by the mock
        handle = mock_open()
        # Get the actual written data
        written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
        # Expected formatted nl string
        from io import StringIO
        buffer = StringIO()
        config = self.config.config
        config.end_comma= True
        config.uppercase = True
        f90nml.write(self.config.config,buffer)
        expected_data = buffer.getvalue()

        self.assertEqual(written_data, expected_data)
        # check backup happened.
        back_file = self.config.filepath()
        back_file = back_file.parent / (back_file.name + '.bak')
        mock_rename.assert_called_once_with(back_file)

    def test_read_value(self):
        # Test the read_value method to ensure it correctly reads a value from the config

        namelist = NamelistVar(type_name='namelist_var', filepath=self.rel_filepath, namelist='NLSTCATM', nl_var='LEXPAND_OZONE')
        value = self.config.read_value(namelist)
        self.assertEqual(value, True)
        # try and read with wrong kind of namelist triggers an error
        namelist = NamelistVar(type_name='json_nl', filepath=self.rel_filepath, namelist='NLSTCATM', nl_var='LEXPAND_OZONE')
        with self.assertRaises(ValueError):
            self.config.read_value(namelist)

    def test_update_value(self):
        # Test the update_value method to ensure it correctly updates a value in the config
        namelist = NamelistVar(type_name='namelist_var', filepath=self.rel_filepath, namelist='namelist', nl_var='var')
        self.config.update_value(namelist, 2,create=True)
        self.assertEqual(self.config.config["namelist"]["var"], 2)
        self.assertTrue(self.config.modified_values[namelist])
        # check get an error if update twice.
        with self.assertRaises(ValueError):
            self.config.update_value(namelist, 3)

class TestUMroseNamelistConfig(unittest.TestCase):
    def setUp(self):
        self.ref_dir = genericLib.expand(
            "$OPTCLIMTOP/OptClimVn3/configurations/example_UM_rose/references/u-db898")
        tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = tmp_dir
        self.root_dir = pathlib.Path(tmp_dir.name)
        shutil.copytree(self.ref_dir, self.root_dir,dirs_exist_ok=True)
        self.rel_filepath = pathlib.Path('app/um/rose-app.conf')
        self.config = UMroseNamelistConfig(root_dir=self.root_dir, rel_filepath=self.rel_filepath)
        self.test_values = {
            ('namelist:clmchfcg','clim_fcg_levls_cfc114'): [-32768.0]*167,
            ('namelist:jules_sea_seaice', 'alpham'): 0.72,
            ('namelist:run_cloud', 'l_add_cca_to_mcica'): True,
            ('namelist:run_radiation', 'h_lwbands'): 9,
        }

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_read(self):
        # Test the read method to ensure it correctly reads namelist data from a file
        config = self.config.read()
        # check  distinct namelists/vars

        for k, v in self.test_values.items():
            print(k)
            self.assertEqual(self.config.parse_value(config.get(k).value), v)
    # rose.config.dump opens a named temp file and writes to that. Then renames the temp file to the tgt.

    def test_write(self):

        # Test the write method to ensure it correctly writes fortran data to a file
        # need to copy the reference and load it.
        with tempfile.TemporaryDirectory() as tmpdir:
            pth = pathlib.Path(tmpdir)
            shutil.copytree(self.root_dir,pth,dirs_exist_ok=True)
            config = UMroseNamelistConfig(root_dir=pth,rel_filepath=self.rel_filepath)
            config.config = metomi.rose.config.ConfigNode() # empty it.
            for k,v in self.test_values.items():
                config.config.set(k,config.to_fortran(v))
            config.write() # write it out.
            # check backup file exists

            back_file = config.filepath()
            back_file = back_file.parent / (back_file.name + '.bak')
            self.assertTrue(back_file.exists())
            # now read config in and check values are as expected.
            new_config = UMroseNamelistConfig(root_dir=pth,rel_filepath=self.rel_filepath)
            for k,v in self.test_values.items():
                got_value = new_config.config.get(k).value
                got_value = new_config.parse_value(got_value)
                self.assertEqual(got_value,v)


    def test_read_value(self):
        # Test the read_value method to ensure it correctly reads a value from the config

        namelist = NamelistVar(type_name='um_rose', filepath=self.rel_filepath,
                               namelist='namelist:run_ozone',
                               nl_var='zon_av_ozone')
        value = self.config.read_value(namelist)
        self.assertEqual(value, False)
        # try and read with wrong kind of namelist triggers an error
        namelist = NamelistVar(type_name='json_nl', filepath=self.rel_filepath, namelist='namelist:run_ozone',
                               nl_var='zon_av_ozone')
        with self.assertRaises(ValueError):
            self.config.read_value(namelist)

    def test_update_value(self):
        # Test the update_value method to ensure it correctly updates a value in the config

        namelist = NamelistVar(type_name='um_rose', filepath=self.rel_filepath, namelist='namelist', nl_var='var')
        self.config.update_value(namelist, 2, create=True)
        got=self.config.read_value(namelist,check_modify=False)
        self.assertEqual(got,2)
        self.assertTrue(self.config.modified_values[namelist])
        # check get an error if update twice.
        with self.assertRaises(ValueError):
            self.config.update_value(namelist, 3)

        # check what happens if update an existing namelist
        namelist2 = NamelistVar(type_name='um_rose', filepath=self.rel_filepath, namelist='namelist:run_radiation',
                                nl_var='two_d_fsd_factor')
        self.config.update_value(namelist2, 1.6)
        # check get failure if value does not exist
        namelist2f = NamelistVar(type_name='um_rose', filepath=self.rel_filepath, namelist='namelist:run_radiation',
                                nl_var='two_d_fsd_factor2')
        with self.assertRaises(KeyError) as error:
            self.config.update_value(namelist2f, 2)
        # check that other values are ok

        # write out
        self.config.write()
        self.config.reset()
        # make sure the value is still there.
        got = self.config.read_value(namelist, check_modify=False)
        self.assertEqual(got, 2)
        # ane make sure another value is also there
        namelist3=NamelistVar(type_name='um_rose', filepath=self.rel_filepath, namelist='namelist:run_cloud', nl_var='allicetdegc')
        got = self.config.read_value(namelist3)
        self.assertEqual(got, -20.0)


class TestGroupConfig(unittest.TestCase):

    def setUp(self):
        # Initialize GroupConfig with a temporary directory
        self.root_dir = pathlib.Path(tempfile.mkdtemp())
        self.group_config = GroupConfig(root_dir=self.root_dir)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.root_dir)

    @patch('pathlib.Path.open', new_callable=mock_open)
    def test_load_config(self, mock_file):
        # Test loading a config for a namelist
        # Will load two configs, one for each namelist
        # After loading, check have two configs, one for each namelist
        group_config = GroupConfig(root_dir=self.root_dir)
        namelist1 = NamelistVar(type_name='json_nl', filepath=pathlib.Path('config1.json'), namelist='namelist1',
                                nl_var='var1')
        namelist2 = NamelistVar(type_name='json_nl', filepath=pathlib.Path('config2.json'), namelist='namelist2',
                                nl_var='var2')
        read_data_list = ['{"namelist1": {"var1": 1}}', '{"namelist2": {"var2": 2}}']
        # Set the side_effect of the mock to return elements from the list
        mock_file.side_effect = [mock_open(read_data=data).return_value for data in read_data_list]
        for nl in [namelist1, namelist2]:
            group_config.load_config(nl)
        self.assertEqual(len(group_config.configs), 2)
        # Check that init_config was called twice
        self.assertEqual(mock_file.call_count, 2)

        # Check that the configs were loaded correctly
        self.assertIn(namelist1.filepath, group_config.configs)
        self.assertIn(namelist2.filepath, group_config.configs)
        self.assertIsInstance(group_config.configs[namelist1.filepath], JSON_Config)
        self.assertIsInstance(group_config.configs[namelist2.filepath], JSON_Config)

    @patch('pathlib.Path.open', new_callable=mock_open, read_data='{"namelist": {"var": 1}}')
    def test_read_value(self, mock_file):
        # Test 1: After running the config should have one element with key from the namelist variable
        group_config = GroupConfig(root_dir=self.root_dir)
        namelist = NamelistVar(type_name='json_nl', filepath=pathlib.Path('config.json'), namelist='namelist',
                               nl_var='var')
        value = group_config.read_value(namelist)
        self.assertEqual(value, 1)

        # Test 2: As Test 1 but with a different namelist variable. Should fail with an error
        namelist_invalid = NamelistVar(type_name='json_nl', filepath=pathlib.Path('config.json'), namelist='namelist',
                                       nl_var='invalid_var')
        with self.assertRaises(KeyError):
            group_config.read_value(namelist_invalid)

        # Test 3: As test 2 but with raise_error = False. Should return None
        value = group_config.read_value(namelist_invalid, raise_error=False)
        self.assertIsNone(value)



    @patch('pathlib.Path.open', new_callable=mock_open, read_data='{"namelist": {"var": 1}}')
    def test_update_value(self, mock_file):
        # Initialize GroupConfig and call update_value
        # Test updating a value in the config
        # Expect config to have one element with key from the namelist filename
        # and the value updated.
        group_config = GroupConfig(root_dir=self.root_dir)
        namelist = NamelistVar(type_name='json_nl', filepath=pathlib.Path('config.json'), namelist='namelist',
                               nl_var='var')

        # Mock the config to have initial value
        group_config.configs[namelist.filepath] = JSON_Config(root_dir=self.root_dir, rel_filepath=namelist.filepath)
        group_config.configs[namelist.filepath].config = {"namelist": {"var": 1}}

        # Update the value
        group_config.update_value(namelist, 2)

        # Assert the value was updated correctly
        self.assertEqual(group_config.configs[namelist.filepath].config["namelist"]["var"], 2)
        self.assertTrue(group_config.configs[namelist.filepath].modified_values[namelist])
        self.assertEqual(len(group_config.configs), 1)

        pass

    @patch('pathlib.Path.open', new_callable=mock_open)
    def test_write_values(self, mock_file):
        # Test writing values to the configs
        # Mock I/O for loading two different configs and writing them out
        # test that configs have length 2 before write_values and 0 after

        # Initialize GroupConfig and load two configs
        group_config = GroupConfig(root_dir=self.root_dir)
        namelists = [
            NamelistVar(type_name='json_nl', filepath=pathlib.Path('config1.json'), namelist='namelist1',
                        nl_var='var1'),
            NamelistVar(type_name='json_nl', filepath=pathlib.Path('config2.json'), namelist='namelist2', nl_var='var2')
        ]

        read_data_list = ['{"namelist1": {"var1": 1}}', '{"namelist2": {"var2": 2}}']
        mock_file.side_effect = [mock_open(read_data=data).return_value for data in read_data_list]
        for nl in namelists:
            group_config.load_config(nl)

        # Check that configs have expected length  before write_values
        self.assertEqual(len(group_config.configs), len(namelists))

        # Mock the write method to avoid actual file I/O
        values = range(1, len(namelists) + 1)
        with patch.object(JSON_Config, 'write', return_value=None) as mock_write:
            group_config.write_values(dict(zip(namelists, values)))

        # Check that configs have length 0 after write_values
        self.assertEqual(len(group_config.configs), 0)

        # Check that the write method was called twice
        self.assertEqual(mock_write.call_count, 2)

    def test_to_dict(self):
        # Test converting GroupConfig to a dictionary
        # Set configs to have two JSON_Config objects. To dict should have two elements
        dct = self.group_config.to_dict()
        self.assertEqual(dct,dict(root_dir=self.root_dir, configs={}))


if __name__ == '__main__':
    unittest.main()
