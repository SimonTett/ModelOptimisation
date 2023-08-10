# test code for Study class.
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdtest
import StudyConfig
import tempfile
from Study import Study
from Model import Model
import pathlib
import os
import generic_json
import importlib.resources




class TestStudy(unittest.TestCase):
    def setUp(self):
        #
        self.test_dir = tempfile.TemporaryDirectory()
        direct = pathlib.Path(self.test_dir.name)
        self.direct = direct
        # Define simulated observations and parameters
        params = {f'param{pcnt}': float(pcnt) for pcnt in range(1, 100)}
        optclim_root = importlib.resources.files("OptClimVn3")
        config = StudyConfig.readConfig(optclim_root / "configurations/dfols14param_opt3.json")
        reference_dir = pathlib.Path(optclim_root).parent / 'Configurations/xnmea'
        self.reference  = reference_dir
        self.config = config
        # Make lots of models
        self.models = []
        for cnt, status in enumerate(list(Model.status_info.keys()) * 2):
            params = {k: float(pcnt) + cnt / 10 for pcnt, k in enumerate(self.config.paramNames())}
            sim_obs = None
            name = f'model{cnt:03d}'
            if status == 'PROCESSED':
                sim_obs = self.fake_fn(params).rename(name)

            model = Model(name, reference=reference_dir,  parameters=params,
                          config_path=direct / (name + '.mcfg'), status=status)
            model.simulated_obs = sim_obs # actually set the simulated obs.
            model.dump_model()
            self.models.append(model)

        # create a study instance
        self.study = Study(config, name="test_study", models=self.models, rootDir=pathlib.Path(direct))

    def fake_fn(self, params):
        import itertools
        obs = self.config.obsNames()
        sim_obs = dict()
        for v, ob in itertools.zip_longest(params.values(), obs, fillvalue=2):
            sim_obs[ob] = v ** 2
        return pd.Series(sim_obs)

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    def test_read_dir(self):
        # read the model configurations from the test_configs directory
        self.study.model_index = dict()
        self.study.read_dir(self.direct)  # read in all the configurations
        # ensure that all the model configurations in the test_configs directory were read
        self.assertEqual(len(self.study.model_index), len(self.models))

    def test_read_configs(self):
        # test that a single model configuration can be read
        study = self.study
        study.model_index = dict()  # no models!
        files = sorted(list(self.direct.glob("**/" + '*.mcfg')))
        models = study.read_model_configs(files[0:1])
        self.assertEqual(models[0].name, 'model000')
        self.assertEqual(list(study.model_index.values())[0].name, 'model000')
        # ensure that the model configuration was added to the models dictionary
        self.assertEqual(len(study.model_index), 1)

    def test_status(self):
        # test that the status method returns a pandas Series object
        status = self.study.status()
        self.assertIsInstance(status, pd.Series)

        # ensure that the index of the Series is the study name
        self.assertEqual(status.name, "test_study")

        # ensure that the values of the Series are the model statuses
        self.assertEqual(status.tolist(), list(Model.status_info.keys()) * 2)

    def test_params(self):
        # test that the params method returns a pandas DataFrame object
        params = self.study.params()
        self.assertIsInstance(params, pd.DataFrame)

        # ensure that the DataFrame has the correct number of rows and columns
        self.assertEqual(params.shape, (len(self.models), len(self.config.paramNames())))

        # test that the normalize option works correctly
        params_norm = self.study.params(normalize=True)
        rng = self.config.paramRanges()
        expected_norm = (params - rng.loc['minParam', :]) / rng.loc['rangeParam', :]
        pdtest.assert_frame_equal(expected_norm, params_norm)

    def test_obs(self):
        # test that the obs method returns a pandas DataFrame object
        obs_scale = self.study.obs()
        self.assertIsInstance(obs_scale, pd.DataFrame)

        # ensure that the DataFrame has the correct number of rows and columns
        self.assertEqual(obs_scale.shape, (2, len(self.config.obsNames())))

        # test that the scale and normalize options work correctly
        obs = self.study.obs(scale=False)
        scale = self.config.scales()
        pdtest.assert_frame_equal(obs * scale, obs_scale)

        obs_norm = self.study.obs(scale=True, normalize=True)
        cov = self.config.Covariances(scale=True)['CovTotal']
        sd = pd.Series(np.sqrt(np.diag(cov)), index=cov.columns)
        tgt = self.config.targets(scale=True)
        pdtest.assert_frame_equal(obs_norm * sd + tgt, obs_scale)

    def test_cost(self):
        # test that the cost method does as expected
        # Not a very satisfactory test as basically reproduces the cost calculation in cost.
        for scale in [False, True]:
            cost = self.study.cost(scale=scale)
            self.assertIsInstance(cost, pd.Series)
            obs = self.study.obs(scale=scale)
            tMat = self.config.transMatrix(scale=scale,
                                           dataFrame=True)  # which puts us into space where totalError is Identity matrix.
            nObs = len(obs.columns)
            tgt = self.config.targets(scale=scale)
            resid = (obs - tgt) @ tMat.T
            cost_expected = np.sqrt((resid ** 2).sum(1).astype(
                float) / nObs)  # TODO -- make nObs the number of indep matrices -- len(resid)
            cost_expected.index = [m.name for m in self.study.model_index.values() if m.status == 'PROCESSED']
            cost_expected = cost_expected.rename(f"cost {self.study.name}")
            pdtest.assert_series_equal(cost_expected, cost)

        # ensure that the Series has the correct number of elements
        self.assertEqual(len(cost), 2)





    def test_key_for_model(self):
        pDict = {'zz': 1.02, 'aa': 1, 'nn': [0, 1]}
        expect = str(('aa', '1', 'nn', '[0, 1]', 'zz', '1.02'))
        model = Model(name='test_model',reference=self.reference,parameters=pDict)
        mkey = self.study.key_for_model(model)
        self.assertEqual(expect,mkey)

    def test_key(self):
        """
        Tests for key
        :return:
        """

        # test key for mixed params is as expected
        pDict = {'zz': 1.02, 'aa': 1, 'nn': [0, 1]}
        expect = str(('aa', '1', 'nn', '[0, 1]', 'zz', '1.02'))
        key = self.study.key(pDict)
        self.assertEqual(key, expect)
        # test that small real differences don't cause any differences.
        pDict = {'zz': 1.0200001, 'aa': 1, 'nn': [0, 1]}
        key = self.study.key(pDict)
        self.assertEqual(key, expect)

    def test_get_model(self):
        """
        Test that model method works
        :return:
        """
        # first case -- one we already have.
        m = list(self.study.model_index.values())[4] # read a model

        m2 = self.study.get_model(m.parameters)
        self.assertEqual(m, m2)

        # and one we don;t have -- should return None
        params=dict(fred=2,harry=3)
        m3 = self.study.get_model(params)
        self.assertIsNone(m3)


if __name__ == '__main__':
    unittest.main()
