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

from namelist_var import namelist_var

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
        filepath =os.environ['OPTCLIMTOP']+'/OptClimVn3/configurations/example_UM_rose/references/u-db898/app/um/rose-app.conf'
        self.config_filepath = pathlib.Path(filepath)
        self.config = metomi.rose.config.load(filepath)
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
        dct_comp.pop('_config_cache')
        self.assertEqual(dct_comp,dct)

    def test_file_cache(self):
        """
        Test file_cache works.
        :return:
        """


        config = self.model.file_cache(self.config_filepath)
        self.assertIsInstance(config,metomi.rose.config.ConfigNode)
        self.assertEqual(config,
                         self.model._config_cache[self.config_filepath])

    def test_write_nml_values(self):
        """
        Test that write_nml_values works!
        :return:
        """
        model = self.model
        model.create_model() # create the model
        nl_info= model.param_info
        pars = nl_info.gen_parameters(self.model,
                                      DP_CORR_STRAT=1e5,AI=0.03)
        model.write_nml_values(pars)
        # read back in the config and check that it only differs
        # for DP_CORR_STRAT & AI
        fname = pars[0][0].filepath
        file = model.model_dir/fname
        # check configs *only* difference are the values of variables changed.
        with open(file,'r+t') as fp:
            config = metomi.rose.config.load(fp)
        file = self.refDir/fname
        with open(file,'r+t') as fp:
            config_o = metomi.rose.config.load(fp)
        self.assertNotEqual(config,config_o)
        # modify config_o to have changed pars and then verify they are the same.
        # Know that filepath is the same so no need to worry about that.
        for p in pars:
            config_o.set(['namelist:'+p[0].namelist,p[0].nl_var],
                         namelist_var.to_fortran(p[1]))
        self.assertEqual(config,config_o)


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

        params_got = self.model.read_params(p)
        self.assertEqual(params_got,self.parameters)

        self.assertEqual(self.model.status,'INSTANTIATED')
        model = self.model.load_model(self.model.config_path)
        # to make sure loaded model is equal need to flush cache.
        #TODO -- have an _eq_ method that does not compare any cache values.
        self.model.clean_cache()
        self.assertEqual(vars(model),vars(self.model ))

    def test_set_params(self ):
        """
        test set_params method
        :return:
        """
        # copy the ref dir into the model_dir
        shutil.copytree(self.refDir,self.model.model_dir,dirs_exist_ok=True)
        self.model.set_params(parameters=dict(AI=1e-2))
        # read in AI from the config
        v = self.model.read_param('AI')
        self.assertEqual(v,1e-2)

    def test_read_nml_var(self):
        """
        test read_nml_var
        :return:
        """

        AI_nl = self.model.param_info.param_constructors['AI'][0]
        shutil.copytree(self.refDir, self.model.model_dir, dirs_exist_ok=True)
        v=self.model.read_nml_var(AI_nl)
        self.assertEqual(v,2.5700e-02)

if __name__ == '__main__':
    unittest.main()
