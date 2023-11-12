import unittest
from archive_study import archive_study
import tarfile
import tempfile
import copy
import pathlib
from Model import Model
from simple_model import simple_model
from SubmitStudy import SubmitStudy
import StudyConfig


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

        submit = SubmitStudy(config, model_name='simple_model', rootDir=testDir, next_iter_cmd=['run myself'])
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
        expected_archive_file = sub.rootDir/(f"archive_{sub.name}.tar") # what the archive_file is called.
        archive_file = arc.archive(sub)
        self.assertEqual(expected_archive_file,archive_file)
        expected_files=[sub.config_path]+[m.config_path for m in sub.model_index.values()]
        expected_files = [pathlib.Path("archive.acfg")]+[pathlib.Path(file).relative_to(sub.rootDir) for file in expected_files]
        with tarfile.open(archive_file,'r') as archive:
            got_files = [pathlib.Path(file) for file in archive.getnames()]
            self.assertEqual(expected_files,got_files)



    def test_extract_archive(self):
        # test extract  method.

        # archive it and then unarchive it to somewhere.
        apth = self.arc.archive(self.submit) # archive everything.
        outdir = self.testDir/'test_archive'
        # and now unarchive it.
        arc,asubmit = self.arc.extract_archive(apth,direct=outdir)
        sub=copy.deepcopy(self.submit)
        sub.rootDir = outdir
        sub.config_path = outdir/sub.config_path.name
        # need to fix the models too!
        for k,m in sub.model_index.items():
            m.config_path = outdir/m.config_path.relative_to(self.submit.rootDir)
            m.model_dir = outdir/m.model_dir.relative_to(self.submit.rootDir)
        self.assertEqual(asubmit,sub)

        # read in an archive generated on Eddie -- tests that remapping happens..
        pth = Model.expand("$OPTCLIMTOP/OptClimVn3/test_data/dfols_r.tar")
        outdir = self.testDir/'test_other_machine'
        arc,submit = self.arc.extract_archive(pth,outdir)
        self.assertIsInstance(submit,SubmitStudy) # should be SubmitStudy
        cost = submit.cost() # and cost should have 51 elements.
        self.assertEqual(len(cost),51)



if __name__ == '__main__':
    unittest.main()
