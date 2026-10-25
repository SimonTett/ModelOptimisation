import unittest

import genericLib
from archive_study import archive_study
import tarfile
import tempfile
import copy
import pathlib
from Model import Model
from SubmitStudy import SubmitStudy
import StudyConfig

genericLib.setup_env()
class TestArchive(unittest.TestCase):

    def setUp(self):
        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(self.tmpDir.name)
        optclim3 = Model.expand('$OPTCLIMTOP/OptClimVn3/')
        refDir = optclim3 / 'configurations/example_Model'
        cpth = refDir / "configurations/dfols14param_opt3.json"
        refDir = refDir / 'reference'
        config = StudyConfig.readConfig(cpth)
        config.baseRunID('ZZ')
        config_path = testDir/'test.cfg'

        submit = SubmitStudy(config, model_name='simple_model', config_path=config_path, next_iter_cmd=['run myself'])
        # create some models
        models=[]
        for param in [dict(VF1=3, CT=1e-4), dict(VF1=2.4, CT=1e-4), dict(VF1=2.6, CT=1e-4)]:
            models.append(submit.create_model(param, dump=True))
        submit.update_iter(models)
        submit.dump_config(dump_models=True)
        self.submit = submit
        self.testDir = testDir
        self.arc =  archive_study()
    def test_archive(self):
        # test archive method.
        # the archive file should contain the config file + N model configs.
        # and the archive file should be called
        arc = self.arc
        sub = self.submit
        expected_archive_file = sub.study_dir / (f"archive_{sub.name}.tar.gz") # what the archive_file is called.
        archive_file = arc.archive(sub)
        self.assertEqual(expected_archive_file,archive_file)
        expected_files=[sub.config_path]+[m.config_path for m in sub.model_index.values()]
        expected_files = sorted([pathlib.Path("archive.acfg")] + [pathlib.Path(file).relative_to(sub.study_dir) for file in expected_files])
        with tarfile.open(archive_file,'r') as archive:
            got_files = sorted([pathlib.Path(file) for file in archive.getnames()])
            self.assertEqual(expected_files,got_files)



    def test_extract_archive(self):
        # test extract  method.

        # archive it and then unarchive it to somewhere.
        apth = self.arc.archive(self.submit) # archive everything.
        outdir = self.testDir/'test_archive'
        # and now unarchive it.
        arc,asubmit = self.arc.extract_archive(apth,direct=outdir)
        sub=copy.deepcopy(self.submit)
        sub.config_path = asubmit.config_path # make sure both studies have the same config path
        # need to fix the models too!
        # means fixing config_patt & history
        for k,m in sub.model_index.items():
            m.config_path = outdir/m.config_path.relative_to(self.submit.study_dir)
            m._history = asubmit.model_index[k]._history
        self.assertEqual(asubmit,sub)

        # read in an archive generated on Eddie -- tests that remapping happens..
        pth = Model.expand("$OPTCLIMTOP/OptClimVn3/test_data/archive_dfols_r.tar")
        outdir = self.testDir/'test_other_machine'
        arc,submit = self.arc.extract_archive(pth,outdir)
        self.assertIsInstance(submit,SubmitStudy) # should be SubmitStudy
        cost = submit.cost() # and cost should have 65 elements.
        self.assertEqual(len(cost),len(submit.model_index))

        # generate an archive with path traversal attack and check it is detected.
        # Detected means both an error on the security log and a ValueError raised.
        # Do by including a file that has a path that goes outside the archive directory.
        evil_dir = self.testDir / 'evil_src'
        evil_dir.mkdir()
        evil_file = evil_dir / 'evil.txt'
        evil_file.write_text('evil')
        # now generate the archive with the evil file in it.
        good_archive = self.arc.archive(self.submit)
        bad_archive = self.testDir / 'bad_archive.tar.gz'
        with tarfile.open(good_archive, 'r') as src, tarfile.open(bad_archive, 'w:gz') as bad:
            for member in src.getmembers():  # copy the good files so the dodgy one is among several
                bad.addfile(member, src.extractfile(member))
            bad.add(evil_file, arcname='../not_really_evil.txt')
            # dodgy path in the archive file would be self.testDir / 'not_really_evil.txt' which is outside  outdir.

        with self.assertLogs(genericLib.secure_logger, level='ERROR') as cm: # secure logger tracks path traversal attacks.
            with self.assertRaises(ValueError): # and get value error.
                self.arc.extract_archive(bad_archive, direct=outdir)
        self.assertTrue(any('Path traversal detected' in line for line in cm.output))
        self.assertFalse((self.testDir / 'not_really_evil.txt').exists())  # nothing written outside outdir





if __name__ == '__main__':
    unittest.main()
