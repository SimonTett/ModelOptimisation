import sys
import unittest
import pathlib
import subprocess
import tempfile
import Model
import platform
import shutil
import genericLib
import json
# Test post-process scripts.

class test_post_process(unittest.TestCase):

    def setUp(self):
        direct = tempfile.TemporaryDirectory()
        self.direct = direct
        self.tempDir = pathlib.Path(direct.name)
        self.script_dir = genericLib.expand("$OPTCLIMTOP/OptClimVn3/Models/post_process_scripts")
        self.config = genericLib.expand("$OPTCLIMTOP/OptClimVn3/configurations/dfols14param_opt3.json")
        self.assertTrue(self.script_dir.exists())  # this should not fail!

    def tearDown(self) -> None:
        """
        Clean up by removing the temp directory contents
        :return:
        """
        shutil.rmtree(self.direct.name, onerror=genericLib.errorRemoveReadonly)
        self.direct.cleanup()

    def test_pp_simple_model(self):
        # test the simple model post processing works
        pth = self.script_dir / "pp_simple_model.py"
        self.assertTrue(pth.exists())
        if platform.system() == 'Windows':
            cmd = [sys.executable, str(pth)]
        else:
            cmd = [str(pth)]
        file_to_copy = self.tempDir/'model_output.json'
        params=dict(vf1=2.2,rhcrit=0.8,kay=100.)
        with open(file_to_copy, 'wt') as fp:
            json.dump(params, fp)

        output_file = 'output.json'
        cmd += [str(self.config),output_file,'-v','-v']
        print(" ".join(cmd))
        stat= subprocess.run(cmd, cwd=self.tempDir, capture_output=True, check=True, text=True)
        print("STDOUT \n",stat.stdout,"="*60)
        print("STDERR \n", stat.stderr, "=" * 60)
        # read in the output and verify it as expected
        with open(self.tempDir/output_file, 'rt') as fp:
            got_params = json.load(fp)

        self.assertEqual(got_params,params)




if __name__ == '__main__':
    unittest.main()
