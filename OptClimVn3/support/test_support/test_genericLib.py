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


        #









if __name__ == '__main__':
    unittest.main()
