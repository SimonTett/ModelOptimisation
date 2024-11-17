import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch, mock_open

import genericLib
from namelist_var import namelist_var, BaseConfig, NamelistVar, JSON_Config, GroupConfig

genericLib.setup_env()


class namelist_var_TestCase(unittest.TestCase):
    def setUp(self):
        """
        Setup for reads. Will have a model + bunch of namelists
        :return:
        """
        # copy reference case to tempdir.

        tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(tmpDir.name)  # used throughout.
        refDir = namelist_var.expand('$OPTCLIMTOP/Configurations/xnmea')  # need a coupled model.
        simObsDir = 'test_in'
        self.dirPath = testDir
        self.refPath = refDir
        self.tmpDir = tmpDir  # really a way of keeping in context
        self.testDir = testDir

        shutil.rmtree(self.testDir, onerror=genericLib.errorRemoveReadonly)
        shutil.copytree(refDir, self.testDir)  # copy everything over.
        nl_list = []
        self.values = [1.0, 10.0,
                       1e-4,
                       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                       3.0]
        for file, nl, var, name in zip(['CNTLATM', 'CNTLATM', 'CNTLATM', 'CNTLATM', 'CNTLATM'],
                                       ['SLBC21', 'RUNCNST', 'SLBC21', 'SLBC21', 'SLBC21'],
                                       ['VF1', 'DTICE', 'CT', 'EACF', 'ENTCOEF'],
                                       ['vf1', 'DTICE', 'ct', 'eacf', 'entcoef']):
            nl = namelist_var(filepath=pathlib.Path(file), namelist=nl, nl_var=var, name=name)
            nl_list.append(nl)
        self.namelist = nl_list

    def tearDown(self):
        shutil.rmtree(self.testDir, onerror=genericLib.errorRemoveReadonly)
        self.tmpDir.cleanup()  # and actually remove it explicitly

    def test_file_cache(self):
        """
        Test file_cache. Read twice get the same result.
        :return:
        """
        nl = self.namelist[0]

        for nl in self.namelist:
            v1 = namelist_var.file_cache(self.dirPath / nl.filepath, clean=True)  # clear cache
            v2 = namelist_var.file_cache(self.dirPath / nl.filepath)  # read (hopefully using the cache)
            self.assertEqual(v1, v2)  #values are the same

    def test_modify_namelists(self):
        """
        Test modify namelists
        :return:
        """
        file_dict = namelist_var.modify_namelists([], dirpath=self.dirPath)  # should be empty
        self.assertEqual(len(file_dict), 0)
        # expect len of unique files
        file_dict = namelist_var.modify_namelists(zip(self.namelist, self.values), dirpath=self.dirPath)
        files = set([nl.filepath for nl in self.namelist])
        self.assertEqual(len(files), len(file_dict))
        for nl in self.namelist:
            self.assertEqual(file_dict[self.dirPath / nl.filepath][nl.namelist][nl.nl_var],
                             nl.read_value(dirpath=self.dirPath))

        file_dict = namelist_var.modify_namelists(zip(self.namelist, [v * 2 for v in self.values]),
                                                  dirpath=self.dirPath,
                                                  update=True)
        # now have set of namelists and values. Check they are as expected.

        for nl in self.namelist:
            self.assertEqual(file_dict[self.dirPath / nl.filepath][nl.namelist][nl.nl_var],
                             nl.read_value(dirpath=self.dirPath) * 2, msg=f"Failed for {nl}")

    def test_nl_modify(self):
        """
        Test namelist modification
        Patch the values. Should have expected bak files and values as expected
        :return:
        """
        values = []
        for v in self.values:
            if isinstance(v, list):
                lst = [vv + 1 for vv in v]
                values.append(lst)
            else:
                values.append(v + 1)
        nl_items = list(zip(self.namelist, values))
        patch = namelist_var.nl_modify(nl_items, dirpath=self.dirPath)
        self.assertTrue(patch)  # worked
        # check namelists are as expected
        for nl, value in nl_items:
            got = nl.read_value(dirpath=self.dirPath)
            self.assertEqual(got, value)

        for nl, v, v2 in zip(self.namelist, self.values, values):
            pth = self.dirPath / nl.filepath
            bak = pth.parent / (pth.name + '.bak')  #check for backup files
            nl2 = namelist_var(filepath=bak.relative_to(self.dirPath), namelist=nl.namelist, nl_var=nl.nl_var)
            self.assertTrue(bak.exists() and bak.is_file())
            self.assertEqual(nl2.read_value(dirpath=self.dirPath), v)
            self.assertEqual(nl.read_value(dirpath=self.dirPath), v2)

    def test_nl_read(self):
        """
        Test reading namelist.
        :return:
        """

        for nl, v in zip(self.namelist, self.values):
            got = nl.read_value(dirpath=self.dirPath)
            self.assertEqual(got, v, msg=f"Failed for {nl}")
        # try again cleaning cache each time. Should get same results
        for nl, v in zip(self.namelist, self.values):
            got = nl.read_value(dirpath=self.dirPath, clean=True)
            self.assertEqual(got, v, msg=f"Failed for {nl}")

    # def test_nl_name(self):
    #     """ Test nl_Name is as expected"""
    #     nl = namelist_var(filepath=pathlib.Path('../test.nl'),namelist='BIG_NL',nl_var='small_var',name='TINY')
    #     self.assertEqual(nl.Name(),'TINY')
    #     nl = namelist_var(filepath=pathlib.Path('../test.nl'),namelist='BIG_NL',nl_var='small_var')
    #     self.assertEqual(nl.Name(),f'{str(nl.filepath)}&BIG_NL small_var')

    def test_repr(self):
        """
        Test representation is as expected.
        :return:
        """
        nl = namelist_var(filepath=pathlib.Path('../test.nl'), namelist='BIG_NL', nl_var='small_var')
        self.assertEqual(nl.__repr__(), f'{str(nl.filepath)}&BIG_NL small_var')

        nl = namelist_var(filepath=pathlib.Path('../test.nl'), namelist='BIG_NL', nl_var='small_var', default=2)
        self.assertEqual(nl.__repr__(), f'{str(nl.filepath)}&BIG_NL small_var default:2')

        nl = namelist_var(filepath=pathlib.Path('../test.nl'), namelist='BIG_NL', nl_var='small_var', default=2,
                          name='TEST')
        self.assertEqual(nl.__repr__(), f'TEST: {str(nl.filepath)}&BIG_NL small_var default:2')


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
        self.config = BaseConfig(root_dir=self.root_dir, rel_filepath=self.rel_filepath, type_name='correct_type')
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
        namelist = NamelistVar(type_name='correct_type', filepath=self.rel_filepath, namelist='test', nl_var='var')
        self.assertTrue(self.config.check_right_nl(namelist))

    def test_check_ok(self):
        # Test case for check_ok method
        # Suggested test case: Check if the method returns True for valid namelist
        namelist = NamelistVar(type_name='correct_type', filepath=self.rel_filepath, namelist='test', nl_var='var')
        self.config.modified_values[namelist] = False
        self.assertTrue(self.config.check_ok(namelist))
        # and Fails for modified
        self.config.modified_values[namelist] = True
        with self.assertRaises(ValueError):
            self.config.check_ok(namelist)

    def test_backup(self):
        # Test case for backup method
        # Suggested test case: Check if the backup file is created correctly
        config = self.config
        backup_path = config.backup(backup=True)
        self.assertTrue(backup_path.exists())

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
