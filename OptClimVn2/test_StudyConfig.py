"""
Test code for StudyConfig
"""

import collections
import os
import platform
import tempfile
import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd
import xarray
import copy

import StudyConfig

os.environ['OPTCLIMTOP'] = os.path.curdir


class testStudyConfig(unittest.TestCase):
    """
    Test cases for StudyConfig. Currently partial so as time passes more tests should be written.
    Ideally >1 for each method
    """

    def setUp(self):
        """
        Standard setup for all test cases
        :return:
        """

        configFile = os.path.join('Configurations', 'example.json')
        self.config = StudyConfig.readConfig(configFile, ordered=True)
        # generate fake -lookup tables
        self.fnLookup = {
            'modelFunction': {'HadCM3': os.getcwd},
            'submitFunction': {'eddie': os.getcwd},
            'optimiseFunction': {'default': os.getcwd},
            'fakeFunction': {'default': os.getcwd}
        }
        # add in some DFOLS info
        dfols = {
            "logging.save_poisedness": False,
            "logging.save_poisedness_comment": "whether or not  to calculate geometry statistics as part of diagnostic information",
            "init.random_initial_directions": True,
            "init.random_initial_directions_comment": "If true perturb in random directions. If true perturb along co-ordinate axis.",
            "noise.additive_noise_level": 0.2,
            "noise.additive_noise_level_comment": "Estimate of noise in cost function. Used in termintion -- nb cost fn is sum of squares **not** sum of squares/nObs",

        }
        self.config.Config['optimise']['dfols'] = {}
        self.config.Config['optimise']['dfols']['namedSettings'] = dfols

    def test_version(self):
        """
        Test version as expected.
        :return: 
        """
        version = self.config.version()
        self.assertEqual(version, 2, msg='Expected version = 2 got %s' % version)

    def test_begin(self):
        """
        Test that begin returns expected values
        :return: 
        """

        expect = pd.Series({"CT": 1.1e-4, "EACF": 0.51, "ENTCOEF": 3.1, "ICE_SIZE": 31e-6, "RHCRIT": 0.71,
                            "VF1": 1.0, "CW_LAND": 2.1e-4})  # values from example.json.
        # first test the parameters are as we expect:
        params = self.config.paramNames()
        params.sort()
        expectKeys = list(expect.keys())
        expectKeys.sort()
        self.assertListEqual(params, expectKeys, msg='Paramters differ')
        got = self.config.beginParam()
        got.sort_values(inplace=True)
        expect.sort_values(inplace=True)
        self.assertEqual(expect.equals(got), True, msg='Differences in beginParam')

        # add test case to deal with scaling within beginParam & nulls (which get standard values)
        Params = self.config.getv('Parameters')  # get the parameters block.
        for p in Params['initParams'].keys():
            Params['initParams'][p] = None  # set value to None
        Params['initScale'] = True
        getParams = self.config.beginParam()
        stdParams = self.config.standardParam()
        self.assertTrue(getParams.equals(stdParams), msg='getParams not stdParams as expected')

        # test that setting works...
        setp = expect * 1.1  # 10% increase
        p = self.config.beginParam(values=setp)
        self.assertTrue(setp.equals(p), msg='getParams not  as expected')
        self.assertFalse(self.config.Config['Parameters']["initScale"])

    def test_Covariances(self):
        """
        test Covariances method.
        :return:
        """
        config = self.config
        cov = config.Covariances()  # example case has constraint on.
        consValue = 2.0 * config.optimise()['mu']
        consName = config.constraintName()
        # expect size nobs X nobs and edge values 0.
        obs = config.obsNames()
        nobs = len(obs)
        expect_slice = np.zeros(nobs)
        covKeys = ['CovTotal', 'CovIntVar', 'CovObsErr']
        for k in covKeys:
            self.assertEqual(cov[k].shape, (nobs, nobs), msg='Shape wrong')
        for k, v in zip(covKeys, [consValue, consValue / 100., consValue]):  # keys and expected value
            expect_slice[-1] = v
            np.testing.assert_array_equal(cov[k].loc[:, consName].values, expect_slice, 'Values wrong -- t2a')
            np.testing.assert_array_equal(cov[k].loc[consName, :].values, expect_slice, 'Values wrong -- t2b')

        # and without constraint
        cov = config.Covariances(constraint=False)  # force constraint off.
        for k in covKeys:
            self.assertEqual(cov[k].shape, (nobs - 1, nobs - 1), msg='Shape wrong without constraint')

        # and test we can overwrite.
        obsNames = config.obsNames()
        c = pd.DataFrame(np.identity(len(obsNames)) * 2, index=obsNames, columns=obsNames)

        cov = config.Covariances(constraint=False, CovTotal=c, CovIntVar=c * 0.1, CovObsErr=c * 0.9)

        cov = config.Covariances()
        cov = config.Covariances(constraint=False, CovTotal=c, CovIntVar=c * 0.1, CovObsErr=c * 0.9)

        for k, scale in zip(covKeys, [1, 0.1, 0.9]):
            self.assertTrue(cov[k].equals(c * scale), msg=f"{k} does not match")

    def test_readCovariances(self):
        """
        Test that readCovariances fails when obs are bad. (and any other tests that seem reasonable)
        :return: nada
        """
        # verify we can read a file OK.
        covInfo =  self.config.getv('study', {}).get('covariance')
        covFile=covInfo['CovObsErr']
        cov=self.config.readCovariances(covFile)
        # now with obsNames not in covariance index.
        with self.assertRaises(ValueError):
            got = self.config.readCovariances(covFile,obsNames=['one','two'])



    def test_version(self):
        """
        Test the version is 2 as expected
        :return: 
        """

        vn = self.config.version()
        self.assertEqual(vn, 2, msg='Got vn = %d expected %d' % (vn, 2))

    def test_referenceConfig(self):
        """
        Test that the ref config path works
        :return: 
        """

        ref = self.config.referenceConfig()
        # expect=os.path.expandvars("$OPTCLIMTOP/Configurations/HadAM3_ed3_SL7_15m")
        expect = os.path.join("Configurations", "HadAM3_ed3_SL7_15m")
        # example json set up for unix so convert any / to \
        if platform.system() == 'Windows':
            ref = ref.replace('/', '\\')
        self.assertEqual(ref, expect, msg='expected %s got %s' % (expect, ref))

        # test we can set it
        expect = 'test'
        self.config.referenceConfig('test')
        self.assertEqual(expect, self.config.referenceConfig())

    def test_cacheFile(self):
        """
        Test that cacheFile method works
        :return: 
        """

        expect = 'cache_file.json'
        got = self.config.cacheFile()
        self.assertEqual(got, expect, msg='cache_file different got %s expected %s' % (got, expect))

    def test_ranges(self):
        """
        Test ranges as expected
        :return: 
        """
        # pNames=self.config.paramNames()
        minP = pd.Series(
            {'CT': 5e-5, 'EACF': 0.5, 'ENTCOEF': 0.6, 'ICE_SIZE': 2.5e-5, 'RHCRIT': 0.6, 'VF1': 0.5, 'CW_LAND': 1e-4},
            name='minParam')
        maxP = pd.Series(
            {'CT': 4e-4, 'EACF': 0.7, 'ENTCOEF': 9.0, 'ICE_SIZE': 4e-5, 'RHCRIT': 0.9, 'VF1': 2.0, 'CW_LAND': 2e-3},
            name='maxParam')
        rng = (maxP - minP).rename('rangeParam', inplace=True)
        got = self.config.paramRanges(minP.index.values)
        self.assertEqual(got.loc['minParam', :].equals(minP), True, msg='Min Values differ')
        self.assertEqual(got.loc['maxParam', :].equals(maxP), True, msg='Max Values differ')
        self.assertEqual(got.loc['rangeParam', :].equals(rng), True, msg='Ranges differ')

    def test_steps(self):
        """
        Test steps are as expected
        :return: 
        """

        step = pd.Series({'CT': 1e-5, 'EACF': 0.02, 'ENTCOEF': 0.15, 'ICE_SIZE': 1.5e-6, 'RHCRIT': 0.01, 'VF1': 0.1,
                          'CW_LAND': 2e-4},
                         name='steps')
        got = self.config.steps(step.index.values)
        self.assertEqual(got.equals(step), True, msg='Step values differ')
        # test that 10% part works -- note that test is specific to vn 2.
        self.config.Config['Parameters']['steps']['VF1'] = None
        rng = self.config.paramRanges(step.index.values)
        step['VF1'] = 0.1 * rng.loc['rangeParam', 'VF1']
        got = self.config.steps(step.index.values)
        self.assertEqual(got.equals(step), True, msg='Step values differ')

        # now add a scaling

        steps = {'CT': 0.1, 'VF1': 0.2, 'scale_steps': True}
        params = ['CT', 'VF1']
        self.config.Config['Parameters']['steps'] = steps
        got = self.config.steps(params)
        rng = self.config.paramRanges(params).loc['rangeParam', :]
        expect = (pd.Series(steps).loc[params] * rng).astype(float).rename(self.config.name())
        self.assertTrue(got.equals(expect))

        # and test changing steps adds values as expected.

        steps = self.config.steps()  # get the steps
        steps.iloc[:-1] = steps.iloc[:-1] * 2  # double some of them
        expect = steps[:]
        steps.loc['scale_steps'] = False
        self.config.steps(steps=steps)  # write them back.
        got = self.config.steps()
        self.assertTrue(got.equals(expect))

    def test_targets(self):
        """
        Test targets are as expected
        :return: 
        """
        # values below are cut-n-paste from some input json file!
        tgt = pd.Series({"rsr_nhx": 102.276779013,
                         "rsr_tropics": 94.172585284,
                         "rsr_shx": 108.113226173,
                         "rsr_gm": 99.5,
                         "olr_nhx": 222.981135462,
                         "olr_tropics": 259.889979681,
                         "olr_shx": 216.123659078,
                         "olr_gm": 239.6,
                         "lat_nhx": 275.837176497,
                         "lat_tropics": 297.551167045,
                         "lat_shx": 287.433252179,
                         "lprecip_nhx": 1.67274541294e-05,
                         "lprecip_tropics": 3.61223235446e-05,
                         "lprecip_shx": 2.23188435704e-05,
                         "mslp_nhx_dgm": 3.30683773931e+02,
                         "mslp_tropics_dgm": 1.78755937185e+02,
                         "temp@500_nhx": 251.407284938,
                         "temp@500_tropics": 266.733035653,
                         "temp@500_shx": 248.927897989,
                         "rh@500_nhx": 53.4221821948,
                         "rh@500_tropics": 33.9426666031,
                         "rh@500_shx": 52.6728738156})
        got = self.config.targets(obsNames=tgt.index.values)
        self.assertEqual(got.equals(tgt), True, msg='Target values differ')
        # test the constraint works too.
        got = self.config.constraintTarget()
        self.assertEqual(got.values, 0.5, msg='Constraint values differ')

        # test we can set value for target.
        set_tgt = pd.Series(dict(obs1=10, obs2=5, obs3=7))
        self.config.obsNames(set_tgt.index.tolist())  # set obsNames to be the target
        self.config.constraint(False)  # do not want constraint
        self.config.targets(targets=set_tgt)  # set the tgt
        got = self.config.targets()
        self.assertTrue(set_tgt.equals(got), 'Target differs when passed in')

        # test that missing an ob causes problems.
        self.config.obsNames(['RSR','olr','olrc'])
        with self.assertRaises(ValueError):
            got = self.config.targets()

    def test_Fixed(self):
        """
        Test can read fixed parameters
        :return: 
        """

        fix = self.config.fixedParams()
        expect = collections.OrderedDict([(u'START_TIME', [1998, 12, 1]), (u'RUN_TARGET', [6, 3, 0])])
        self.assertEqual(expect, fix, msg='fix not as expected')
        # test cases with none work.
        values = self.config.Config['Parameters']['fixedParams']
        values['VF1'] = None  # should overwrite values in array.
        values['CW_LAND'] = None  # as above
        values['ALPHAM'] = None
        values['NoSuchParam'] = None
        fix = self.config.fixedParams()
        self.assertEqual(fix['VF1'], 1.0, msg='VF1 not as expected')  # should be set to default value
        self.assertEqual(fix['CW_LAND'], 2e-4, msg='CW_LAND not as expected')  # should be set to default value
        self.assertEqual(fix['ALPHAM'], 0.5, msg='ALPHAM not as expected')  # should be set to default value
        self.assertIsNone(fix['NoSuchParam'], msg='NoSuchParam should be None')  # should be None
        # set values['VF1'] to 2 and check it is still 2.
        values['VF1'] = 2.0
        fix = self.config.fixedParams()
        self.assertEqual(fix['VF1'], 2.0, msg='VF1 not as expected 2nd time')

    def test_runTime(self):
        """
        Test that runtime is None
        :return: 
        """
        self.assertIsNone(self.config.getv("runTime"))

    def test_modelFunction(self):
        """
        test the modelFunction works as expected
        :return:
        """

        fn = self.config.modelFunction(self.fnLookup['modelFunction'])
        expect = self.fnLookup['modelFunction']['HadCM3']
        self.assertEqual(fn, expect)

    def test_submitFunction(self):
        """
        test the modelFunction works as expected
        :return:
        """

        fn = self.config.submitFunction(self.fnLookup['submitFunction'])
        expect = self.fnLookup['submitFunction']['eddie']
        self.assertEqual(fn, expect)

    def test_optimiseFunction(self):
        """
        test the optimiseFunction works as expected
        :return:
        """

        fn = self.config.optimiseFunction(self.fnLookup['optimiseFunction'])
        expect = self.fnLookup['optimiseFunction']['default']
        self.assertEqual(fn, expect)

    def test_fakeFunction(self):
        """
        test the optimiseFunction works as expected
        :return:
        """

        fn = self.config.optimiseFunction(self.fnLookup['fakeFunction'])
        expect = self.fnLookup['fakeFunction']['default']
        self.assertEqual(fn, expect)

    def test_optimumParam(self):
        """
        Test that optimum param works.
        :return:
        """
        # nothing there to start with so should get array of nans back.

        optParam = self.config.optimumParams()
        self.assertEqual(np.sum(optParam.isnull()), len(optParam), msg='optParam should  be all Nan')
        # now set it
        # need to set parameter list.
        res = self.config.optimumParams(RHCRIT=2.1, VF1=2.005, ENTCOEF=3.2)
        expect = pd.Series(
            {'VF1': 2.005, 'ENTCOEF': 3.2, 'RHCRIT': 2.1, 'CT': 1e-4, 'EACF': 0.5, 'ICE_SIZE': 30e-6, 'CW_LAND': 2e-4})
        expect = expect[self.config.paramNames()]
        self.assertTrue(res.equals(expect))

    def test_obsNames(self):
        """
        Test obsNames works
        :return:
        """

        expect = self.config.getv('study', {}).get('ObsList', [])[:]  # copy the Obslist (as modify expect later)
        # first with constraint Turned off
        obs = self.config.obsNames(add_constraint=False)
        self.assertEqual(obs, expect, msg='Constraint off failed')

        # then with constraint on.
        expect.append(self.config.constraintName())
        obs = self.config.obsNames()  # and with constraint included.
        self.assertEqual(obs, expect, msg='Constraint on failed')
        # and set names.
        newNames = ['obs1', 'obs2', 'obs3']
        got = self.config.obsNames(obsNames=newNames)
        got = self.config.obsNames() # got will include the constraint name here
        newNames.append(self.config.constraintName())
        self.assertEqual(got, newNames)  # we should get the names back

    def test_paramNames(self):
        """
        test paramNames

        """

        got = self.config.paramNames()  # should get them back..
        begin = self.config.beginParam()  # paramnames come from init valye
        expect = begin.index.tolist()
        self.assertEqual(expect, got, msg="params not as expected")

    def test_GNsetget(self):
        """
        test that GNsetget works as expects
        :return:
        """
        import numpy as np
        # first case -- nothing set then should get None back and self.getv('GNinfo') should return None
        self.config.GNgetset('slartybartfast')
        self.assertIsNone(self.config.getv('GNinfo'))
        value = np.arange(1.0, 20.0)
        got = self.config.GNgetset('slartybartfast', value)
        self.assertTrue(np.array_equal(value, got))
        # different value works
        value = value * 2
        got = self.config.GNgetset('slartybartfast', value)
        self.assertTrue(np.array_equal(value, got))
        # no having set it should just get it back.
        got = self.config.GNgetset('slartybartfast')
        self.assertTrue(np.array_equal(value, got))

        # self.fail()

    def test_GNjacobian(self):
        """
        test that GNjcobian works.
        :return:
        """

        # nothing to start with -- should get None
        self.assertIsNone(self.config.GNjacobian(), msg='Jacobian should  be None')
        # now check we get an xarray back with expected values.
        paramNames = self.config.paramNames()
        obsNames = self.config.obsNames(add_constraint=True)
        jac = np.random.uniform(0, 1, (2, len(paramNames), len(obsNames)))  # random numpy array
        expect = xarray.DataArray(jac, coords={'Iteration': np.arange(0, 2), 'Parameter': paramNames,
                                               'Observation': obsNames},
                                  dims=['Iteration', 'Parameter', 'Observation'], name=self.config.name())
        jacx = self.config.GNjacobian(jac, constraint=True)
        self.assertTrue(jacx.equals(expect), msg='failed in GNjacobian')
        # and now it is set should get back same results.
        jacx = self.config.GNjacobian(constraint=True)
        self.assertTrue(jacx.equals(expect), msg='failed in GNjacobian#2')

    def test_GNparams(self):
        """
        Test that GNparams works as expected.
        :return:
        """

        # nothing to start with -- should get None
        self.assertIsNone(self.config.GNparams(), msg='Params should  be None')
        # now check we get an xarray back with expected values.
        paramNames = self.config.paramNames()
        par = np.random.uniform(0, 1, (2, len(paramNames)))  # random numpy array
        expect = xarray.DataArray(par, coords={'Iteration': np.arange(0, 2), 'Parameter': paramNames},
                                  dims=['Iteration', 'Parameter'], name=self.config.name())
        parx = self.config.GNparams(par)
        self.assertTrue(parx.equals(expect), msg='failed in GNparams')
        # and now it is set should get back same results.
        parx = self.config.GNparams()
        self.assertTrue(parx.equals(expect), msg='failed in GNparams#2')

    def test_GNhessian(self):
        """
        Test that GNhessian works as expected
        :return:
        """

        # nothing to start with -- should get None
        self.assertIsNone(self.config.GNhessian(), msg='Hessian should  be None')
        # now check we get an xarray back with expected values.
        paramNames = self.config.paramNames()
        hes = np.random.uniform(0, 1, (2, len(paramNames), len(paramNames)))  # random numpy array
        expect = xarray.DataArray(hes, coords={'Iteration': np.arange(0, 2), 'Parameter': paramNames,
                                               'Parameter_2': paramNames},
                                  dims=['Iteration', 'Parameter', 'Parameter_2'], name=self.config.name())
        hesx = self.config.GNhessian(hes)
        self.assertTrue(hesx.equals(expect), msg='failed in GNhessian')
        # and now it is set should get back same results.
        hesx = self.config.GNhessian()
        self.assertTrue(hesx.equals(expect), msg='failed in GNhessian#2')

    def test_GNcost(self):
        """
        Test that GNcost works as expected
        :return:
        """
        # nothing to start with -- should get None
        self.assertIsNone(self.config.GNcost(), msg='Cost should  be None')
        # now check we get an xarray back with expected values.
        cost = np.random.uniform(0, 1, (2))  # random numpy array
        expect = pd.Series(cost, index=np.arange(0, 2), name=self.config.name())
        expect.index.rename('Iteration', inplace=True)
        costx = self.config.GNcost(cost)
        self.assertTrue(costx.equals(expect), msg='failed in GNcost')
        # and now it is set should get back same results.
        costx = self.config.GNcost()
        self.assertTrue(costx.equals(expect), msg='failed in GNcost#2')

    def test_GNalpha(self):
        """
        Test that GNalpha works as expected
        :return:
        """
        # nothing to start with -- should get None
        self.assertIsNone(self.config.GNalpha(), msg='alpha should  be None')
        # now check we get an pd.Series back with expected values.
        alpha = np.random.uniform(0, 1, (2))  # random numpy array
        expect = pd.Series(alpha, index=np.arange(0, 2), name=self.config.name())
        expect.index.rename('Iteration', inplace=True)
        alphax = self.config.GNalpha(alpha)
        self.assertTrue(alphax.equals(expect), msg='failed in GNalpha')
        # and now it is set should get back same results.
        alphax = self.config.GNalpha()
        self.assertTrue(alphax.equals(expect), msg='failed in GNalpha#2')

    def test_standardParam(self):
        """
        Test that standardParam works as expected.

        :return:
        """
        expect = pd.Series({"CT": 1e-4, "EACF": 0.5, "ENTCOEF": 3.0, "ICE_SIZE": 30e-6, "RHCRIT": 0.7,
                            "VF1": 1.0, "CW_LAND": 2e-4})

        expect = pd.Series(collections.OrderedDict(
            (("CT", 1e-4), ("EACF", 0.5), ("ENTCOEF", 3.0), ("ICE_SIZE", 30e-6), ("RHCRIT", 0.7),
             ("VF1", 1.0), ("CW_LAND", 2e-4))))
        params = self.config.standardParam()
        self.assertTrue(params.equals(expect), msg='Params not as expected')
        # test set value works as expected..
        expect.CT = 2.0
        params = self.config.standardParam(expect)
        self.assertTrue(params.equals(expect), msg='Params not as expected when setting')

    def test_ensembleSize(self):
        """
        test that ensembleSize works
        :return:
        """

        nens = self.config.ensembleSize()
        self.assertEqual(nens, 1)
        # now set it
        self.config.ensembleSize(10)
        nens = self.config.ensembleSize()
        self.assertEqual(nens, 10)

    def test_constraint(self):
        """
        test that constraint works
        :return:
        """
        constraint = self.config.constraint()
        self.assertTrue(constraint, 'Constraint not True')
        # now set it
        self.config.constraint(False)
        constraint = self.config.constraint()
        self.assertFalse(constraint, 'Constraint not False')

    def test_baseRunID(self):
        """
        Test that baseRunID works
        :return:
        """

        self.assertEqual(self.config.baseRunID(), 'zz')  # get what we expect
        self.config.baseRunID('xx')  # set it
        self.assertEqual(self.config.baseRunID(), 'xx')  # and check it set.

    def test_maxDigits(self):
        """

        test that maxDigits works

        Two cases:
        1) Not set (default) -- should return None
        2) Set it -- should get value back.

        """

        got = self.config.maxDigits()
        self.assertIsNone(got, 'Expected None')
        self.config.maxDigits(2)
        expect = 2
        got = self.config.maxDigits()
        self.assertEqual(expect, got, f'Expected {expect} got {got} ')

    def test_copy(self):
        """
        Test that copy works
        :return:
        """

        copy = self.config.copy()
        self.assertNotEqual(copy._filename,
                            self.config._filename)  # should be different because filenames are different
        copy._filename = self.config._filename
        self.assertEqual(copy.Config, self.config.Config)

    def test_plot(self):
        """
        Test that plot works
        :return:
        """
        tfile = tempfile.NamedTemporaryFile(suffix='.png')
        file = tfile.name
        self.config.plot(monitorFile=file)
        # check file exists
        self.assertTrue(os.path.isfile(file))
        # file should be deleted when tfile goes out of scope

    def test_transMatrix(self):
        """
        test transformation functionality
        :return: nada
        """

        for scale in [True, False]:
            tMat = self.config.transMatrix(scale=scale)
            cov = self.config.Covariances(scale=scale)['CovTotal']
            got = tMat.dot(cov).dot(tMat.T)
            expect = np.identity(got.shape[0])  # trans matrix might, in effect, truncate matrix.
            atol = 1e-7
            rtol = 1e-7

            nptest.assert_allclose(got, expect, atol=atol, rtol=rtol,
                                   err_msg=f' Scale {scale} Transform not giving I')

    def test_DFOLS_userParams(self):
        """
        test DFOLS_userParams.
        :return:  nada
        :return:  nada
        """
        import re

        # test case
        dfols = {  # cut-n-paste from setup
            "logging.save_poisedness": False,
            "logging.save_poisedness_comment": "whether or not  to calculate geometry statistics as part of diagnostic information",
            "init.random_initial_directions": True,
            "init.random_initial_directions_comment": "If true perturb in random directions. If true perturb along co-ordinate axis.",
            "noise.additive_noise_level": 0.2,
            "noise.additive_noise_level_comment": "Estimate of noise in cost function. Used in termintion -- nb cost fn is sum of squares **not** sum of squares/nObs",

        }
        dfolsConfig = self.config.DFOLS_userParams()
        self.assertEqual(len(dfolsConfig), 3, 'Expecting 3 elements in dfolsConfig')
        # and verify that values (after stripping comments are good)
        expect = {}
        for k, v in dfols.items():
            if not re.search(r'_comment\s*$', k):
                expect[k] = v
        self.assertEqual(dfolsConfig, expect, 'config not as expected')

        # case where we pass in some values
        newV = {'randomParm': 2, 'noise.additive_noise_level': 20.}
        dfolsConfig = self.config.DFOLS_userParams(newV)  # note that noise.additive_noise_level will be overwritten.
        expect.update(randomParm=2)  # update expect
        self.assertEqual(dfolsConfig, expect, 'config not as expected')
        # check that updateParams works
        expect.update(newV)  # all new value
        self.config.DFOLS_userParams(updateParams=expect)

        self.assertEqual(expect, self.config.DFOLS_userParams(), 'user params not as expected')

        # check IDs are different for different calls to config

        self.assertNotEqual(id(self.config.DFOLS_userParams()), id(self.config.DFOLS_userParams()))

    def test_DFOLS_config(self):
        """
        Test DFOLS_config

        """

        config = copy.deepcopy(self.config.DFOLS_config())
        # should be the same as the raw data
        dfols = self.config.optimise()['dfols']
        self.assertEqual(config, dfols)
        # modify dfols set it and check.
        config['rhobeg'] = 0.1
        # underlying data should be different
        self.assertNotEqual(config, self.config.DFOLS_config())
        self.config.DFOLS_config(config)  # set it and now should be the same
        self.assertEqual(config, self.config.DFOLS_config())

    def test_dataFrameInfo(self):

        """
        Test can set/get dataFrameInfo.

        """
        import string
        df = pd.DataFrame(np.random.uniform(0, 1, [20, 20]))
        self.config.set_dataFrameInfo(randomMatrix=df)
        got = self.config.get_dataFrameInfo('randomMatrix')
        atol = 1e-10
        nptest.assert_allclose(got, df,
                               atol=atol)  # round tripping losing some precision. (fp conversion does not quite go to full precision)
        # now add another two and get them all back.
        self.config.set_dataFrameInfo(twoRandom=2 * df, minusRandom=-df)
        got1, got2, got3 = self.config.get_dataFrameInfo(['randomMatrix', 'twoRandom', 'minusRandom'])
        nptest.assert_allclose(got1, df, atol=atol)
        nptest.assert_allclose(got2, 2 * df, atol=atol)
        nptest.assert_allclose(got3, -df, atol=atol)
        # and try with a pandas series.
        nindex = 5
        ser = pd.Series(np.arange(nindex), list(string.ascii_letters[0:nindex]))
        self.config.set_dataFrameInfo(ser=ser)
        get = self.config.get_dataFrameInfo('ser')
        nptest.assert_allclose(ser,get)

    def test_transJacobian(self):
        """
        Test transJacobian method works

        """

        jac = pd.DataFrame(np.identity(5))
        jac.iloc[0, 1] = 0.0001
        self.config.transJacobian(jac)
        got = self.config.transJacobian()
        self.assertTrue(jac.equals(got))

    def test_jacobian(self):
        """
        Test jacobian method works

        """

        jac = pd.DataFrame(np.identity(5))
        jac.iloc[0, 1] = 0.0001
        self.config.jacobian(jac)
        got = self.config.jacobian()
        self.assertTrue(jac.equals(got))

    def test_hessian(self):
        """
        Test hessian method works

        """

        hessian = pd.DataFrame(np.identity(5))
        hessian.iloc[0, 1] = 0.0001
        self.config.hessian(hessian)
        got = self.config.hessian()
        self.assertTrue(hessian.equals(got))

    def test_optimise(self):
        """
        Test can get back the whole optimise object (and set it)
        :return:
        """

        opt = self.config.optimise() # should be a dict
        self.assertEqual(type(opt),dict)
        # set a value
        opt2 = self.config.optimise(maxIterations=10)
        self.assertEqual(opt2['maxIterations'],10) # value as expected
        opt3 = self.config.optimise() # get it back. Should have changed.
        self.assertEqual(opt2['maxIterations'], 10)

    def test_scales(self):
        """
        Test scaling works as expected
        Tests:
            1) get expected values.  1 for values not specified. actaul value for values specified
            2) Can set values. With 1's ignored.
            :return:nada
        """

        expected =pd.Series({key:self.config.Config['scalings'].get(key,1.0) for key in self.config.obsNames()})
        got = self.config.scales()
        nptest.assert_array_equal(expected,got)
        # set some values
        test_scales = dict(one=2,two=1,three=4)
        obsNames= test_scales.keys()
        expect = pd.Series(test_scales)
        # get an error because obsNames differ.
        with self.assertRaises(ValueError):
            scales = self.config.scales(test_scales)
        scales = self.config.scales(test_scales,obsNames=obsNames)
        nptest.assert_array_equal(scales,expect)
        scales = self.config.scales(dict(one=2,three=4),obsNames=obsNames)
        nptest.assert_array_equal(scales, expect)

if __name__ == "__main__":
    print("Running Test Cases")
    unittest.main()  ## actually run the test cases
