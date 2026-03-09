import pathlib
import tempfile
import unittest
import genericLib

class genericLib_test(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory before each test
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = pathlib.Path(self.tmpdir.name)

    def tearDown(self):
        # Cleanup the temporary directory after each test
        self.tmpdir.cleanup()
    def test_parse_isoduration(self):
        """
        Test parse_isoduration

        :return: Nada. Just runs tests.
        """

        strings = ['P1Y1M1DT1H1M1.11S', 'P2Y', 'PT1M', 'PT1M2.22S', 'P-1Y']
        expected = [[1, 1, 1, 1, 1, 1.11], [2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 2.22],
                    [-1, 0, 0, 0, 0, 0]]

        for s, e in zip(strings, expected):
            got = genericLib.parse_isoduration(s)
            self.assertEqual(e, got)
            # now reverse
            got = genericLib.parse_isoduration(e)
            self.assertEqual(got, s)

        # should fail without a leading P or not a string
        for fail_case in ['12Y', '1Y', '3MT2S']:
            with self.assertRaises(ValueError):
                got = genericLib.parse_isoduration(fail_case)
            #

    def test_seconds_to_isodurtion(self):
        # test seconds to isoduration
        # test cases

        test_cases = [
            (3661.11, 'PT1H1M1.110S'),
            (60, 'PT1M'),
            (3600, 'PT1H'),
            (86400, 'P1D'),
            (31536000, 'P365D')
        ]

        for seconds, expected in test_cases:
            got = genericLib.seconds_to_isoduration(seconds)
            self.assertEqual(expected, got)
            # test reversability
            reverse = genericLib.parse_isoduration(got)
            # need to convert to seconds..
            rev_seconds =0.0
            for rev,conv in zip(reverse, [31536000, 2592000, 86400, 3600, 60, 1]):
                rev_seconds += rev * conv

            self.assertEqual(seconds, rev_seconds,msg=f'{got} failed for {seconds} with {reverse}')

        # try -ve seconds which should fail
        with self.assertRaises(ValueError):
            got = genericLib.seconds_to_isoduration(-1.00)


    def test_likely_text_file(self):
        # test likely text file
        file_path = self.tmp_path/'example.txt'

        with file_path.open('wt') as file:
            file.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit.')
        self.assertTrue(genericLib.likely_text_file(file_path))
        # test a binary file
        # Generate a binary file with non-text content
        file_path = self.tmp_path/'binary_file.bin'

        with file_path.open('wb') as binary_file:
            # Write some arbitrary binary data
            binary_file.write(b'\x00\xFF\x10\x20\x30\x40\x50\x60\x70\x80\x90\xA0\xB0\xC0\xD0\xE0\xF0')
        self.assertFalse(genericLib.likely_text_file(file_path))

    def test_backup_file(self):
        # test backup file
        # need a tempdir


        # Case 1 -- file name and backup name is file name + .bak
        file = self.tmp_path/"example.txt"
        expected_backup = self.tmp_path/"example.txt.bak"
        # put some text into example.txt
        with file.open('w') as f:
            f.write('This is a test file.')
        got_backup = genericLib.backup_file(file)
        self.assertEqual(expected_backup, got_backup)
        # put some text into example.txt
        with file.open('w') as f:
            f.write('This is a test file.')
        # now move the input file.
        got_backup = genericLib.backup_file(file,create='move')
        self.assertTrue(got_backup.exists())
        self.assertFalse(file.exists())
        with self.assertRaises(FileNotFoundError):
            got_backup = genericLib.backup_file(file, create='move')
        # npw move the backup file back to the original name and copy it
        got_backup.rename(file)
        got_backup = genericLib.backup_file(file, create='copy')
        self.assertTrue(got_backup.exists())
        self.assertTrue(file.exists())
        # test that making a backup when the file exists gives None
        with self.assertLogs(genericLib.my_logger, level='WARNING') as log:
            got_backup = genericLib.backup_file(file, create='copy')
        self.assertIsNone(got_backup)


    def test_copy_files(self):
        # test copy_files works
        init_dir = self.tmp_path/'init'
        dest_dir = self.tmp_path/'dest'
        init_dir.mkdir()
        files=[pathlib.Path(f) for f in ['file1.txt', 'file2.txt', 'file3.txt']]
        # test symlink False & True
        for symlink in [False, True]:
            dest_subdir = dest_dir/f'symlink_{symlink}'

            for file in files:
                with (init_dir/file).open('w') as f:
                    f.write(f'This is {file}')
            genericLib.copy_files(init_dir, dest_dir, files, symlinks=symlink)
            for file in files:
                self.assertTrue((dest_dir/file).exists())
                # test file contents are identical
                with (dest_dir/file).open('r') as f:
                    dcontent = f.read()
                with (init_dir/file).open('r') as f:
                    icontent = f.read()
                self.assertEqual(icontent, dcontent, msg=f'File contents differ for {file} with symlink={symlink}')


    def test_expand(self):
        # Test expansion of environment variable
        # AI generated after review of AI proposed tests. Code being tested written by human
        import os
        env_var = 'HOME'
        os.environ[env_var] = str(self.tmp_path)
        result = genericLib.expand(f"${env_var}/file.txt")
        self.assertEqual(result, self.tmp_path/'file.txt')

        # Test expansion of user home
        result = genericLib.expand("~/file.txt")
        self.assertTrue(str(result).endswith('file.txt'))

        # Test with a string that has no variables
        result = genericLib.expand("plain.txt")
        self.assertEqual(result, pathlib.Path("plain.txt"))

        # Test error handling: unexpanded env var, error='fail'
        with self.assertRaises(ValueError):
            genericLib.expand("$UNSET_VAR/file.txt", error='fail')

        # Test error handling: unexpanded env var, error='warn'
        with self.assertLogs(genericLib.my_logger, level='WARNING'):
            result = genericLib.expand("$UNSET_VAR/file.txt", error='warn')
            self.assertIn("$UNSET_VAR", str(result))

        # Test error handling: unexpanded env var, error='ignore'
        result = genericLib.expand("$UNSET_VAR/file.txt", error='ignore')
        self.assertIn("$UNSET_VAR", str(result))

        # Test Windows-style env variable expansion: should raise, warn, or ignore
        with self.assertRaises(ValueError):
            genericLib.expand("%FOO%/baz.txt", error='fail')

        with self.assertLogs(genericLib.my_logger, level='WARNING'):
            result = genericLib.expand("%FOO%/baz.txt", error='warn')
            self.assertIn("%FOO%", str(result))

        result = genericLib.expand("%FOO%/baz.txt", error='ignore')
        self.assertIn("%FOO%", str(result))

        # Test with empty string
        result = genericLib.expand("")
        self.assertEqual(result, pathlib.Path(""))

        # Test with a string containing only env variable
        os.environ['MYVAR'] = 'myvalue'
        result = genericLib.expand("$MYVAR")
        self.assertEqual(result, pathlib.Path("myvalue"))

        # Test with a string containing only user home
        result = genericLib.expand("~")
        self.assertTrue(str(result).endswith(str(pathlib.Path.home())))

    def test_error_handle(self):
        # Test error_handle raises ValueError when mode is 'fail'
        with self.assertRaises(ValueError) as cm:
            genericLib.error_handle('This is a fail message','fail', )
        self.assertIn('This is a fail message', str(cm.exception))

        # Test error_handle logs a warning and returns gracefully when mode is 'warn'
        with self.assertLogs(genericLib.my_logger, level='WARNING') as log:
            result = genericLib.error_handle('This is a warn message','warn', )
            self.assertIsNone(result)
            self.assertTrue(any('This is a warn message' in record for record in log.output))

        # Test error_handle ignores the error and returns gracefully when mode is 'ignore'
        result = genericLib.error_handle( 'This is an ignore message','ignore')
        self.assertIsNone(result)

        # Test error_handle with empty message
        with self.assertRaises(ValueError):
            genericLib.error_handle('','fail')
        with self.assertLogs(genericLib.my_logger, level='WARNING') as log:
            genericLib.error_handle('', 'warn')
        result = genericLib.error_handle('','ignore')
        self.assertIsNone(result)

        # Test error_handle with invalid mode (should raise ValueError)
        with self.assertRaises(ValueError):
            genericLib.error_handle('invalid_mode', 'Invalid mode message')

    def notest_setup_logging(self):
        """
        Test that setup_logging configures logging correctly with various parameters and handles multiple calls without duplication.
          Also tests error handling for invalid log levels and log configurations.
          AI generated and human reviewed on strategy and code. Code being tested written by human. code review cursory but some testing better than none.
          Causes some persistent logger issues that caues test_param_info.test_register and test_param_info.test_update_from_file to fail.
          At some future point this method should be rewritten to reset logs.
        :return:
        """
        import logging
        # Default setup: should allow info messagesI
        genericLib.setup_logging()
        with self.assertLogs(logging.getLogger(), level='INFO') as log:
            logging.info('Default info message')
            self.assertTrue(any('Default info message' in record for record in log.output))

        # Custom log level: DEBUG
        genericLib.setup_logging(level='DEBUG',rootname='OPTCLIM.genericLib')
        with self.assertLogs(logging.getLogger(), level='DEBUG') as log:
            logging.debug('Debug message')
            self.assertTrue(any('Debug message' in record for record in log.output))

        # Custom log level: WARNING
        genericLib.setup_logging(level='WARNING',rootname='OPTCLIM.genericLib')
        with self.assertLogs(logging.getLogger(), level='WARNING') as log:
            logging.warning('Warning message')
            self.assertTrue(any('Warning message' in record for record in log.output))
        # Info/debug should not appear
        logging.info('Should not appear')
        logging.debug('Should not appear')

        # Custom root logger name
        genericLib.setup_logging(rootname='customLogger')
        logger = logging.getLogger('customLogger')
        with self.assertLogs(logger, level='INFO') as log:
            logger.info('Custom logger info')
            self.assertTrue(any('Custom logger info' in record for record in log.output))

        # Multiple calls: should not duplicate logs
        genericLib.setup_logging()
        genericLib.setup_logging()
        with self.assertLogs(logging.getLogger(), level='INFO') as log:
            logging.info('No duplicate message')
            self.assertTrue(any('No duplicate message' in record for record in log.output))

        # Invalid log level: should raise ValueError or handle gracefully
        with self.assertRaises(ValueError):
            genericLib.setup_logging(level='INVALID')

        # Log config passed: valid config
        log_config = {
            'version': 1,
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                }
            },
            'root': {
                'handlers': ['console'],
                'level': 'INFO',
            }
        }
        genericLib.setup_logging(log_config=log_config,level='INFO')
        with self.assertLogs(logging.getLogger(), level='INFO') as log:
            logging.info('Log config info')
            self.assertTrue(any('Log config info' in record for record in log.output))

        # Log config passed: invalid config
        bad_config = { 'root': {}}  # missing version. Should raise an error
        with self.assertRaises(Exception):
            genericLib.setup_logging(log_config=bad_config,level='INFO')

        # and reset logging to default for other tests
        genericLib._reset_all_loggers()
        for name, logger in logging.root.manager.loggerDict.items():
            print(
                f"Logger: {name}, Handlers: {getattr(logger, 'handlers', None)}, Level: {getattr(logger, 'level', None)}")

if __name__ == '__main__':
    unittest.main()
