import os
import unittest

import metomi.rose.config
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
            DP_CORR_STRAT=500.0,TWO_D_FSD_FACTOR=2,
            ENT_FAC_DP= 1.0, AI=3e-2
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
                            model_dir=testDir, post_process=post_process,
                            parameters=parameters)
        self.config_path = self.model.config_path

        shutil.copy(simObsDir / '01_GN' / 'h0101' / 'observables.nc',
                    testDir / 'obs.nc')  # copy over a netcdf file of observations.

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
        shutil.copytree(self.refDir,self.model.model_dir,dirs_exist_ok=True)
        val=self.model.read_param('AI')
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

    def test_set_params(self ):
        """
        test set_params method
        :return:
        """
        # copy the ref dir into the model_dir
        shutil.copytree(self.refDir,self.model.model_dir,dirs_exist_ok=True)
        params = dict(AI=1e-2,DP_CORR_STRAT=500.0,TWO_D_FSD_FACTOR=2,ENT_FAC_DP= 1.0)
        self.model.set_params(parameters=params)
        # test values
        for p,v in params.items():
            self.assertEqual(v,self.model.read_param(p))




if __name__ == '__main__':
    unittest.main()
