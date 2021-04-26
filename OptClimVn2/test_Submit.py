"""
Place to put tests for Submit.
"""
import collections
import copy
import os
import shutil
import tempfile
import unittest

import numpy as np
import numpy.testing as nptest
import pandas as pd

import config  # need the configuration.
from OptClimVn2 import Submit, optClimLib, ModelSimulation, StudyConfig


class testSubmit(unittest.TestCase):
    """
    Test cases for Submit There should be one for every method in Submit.

    """

    def setUp(self):
        """
        Standard setup for all test cases
        :return:
        """
        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = self.tmpDir.name
        refDir = 'test_in'

        testDir = os.path.expanduser(os.path.expandvars(testDir))
        refDir = os.path.expandvars(os.path.expanduser(refDir))
        if os.path.exists(testDir):  # remove directory if it exists
            shutil.rmtree(testDir, onerror=optClimLib.errorRemoveReadonly)
        self.dirPath = testDir
        self.refPath = refDir
        refDirPath = os.path.join(refDir, 'start')
        # now do stuff.
        # create a ModelSubmit instance... and then read in dir.

        jsonFile = os.path.join('Configurations', 'example.json')
        configData = StudyConfig.readConfig(filename=jsonFile, ordered=True)  # parse the jsonFile.
        begin = configData.beginParam()
        keys = begin.keys()
        keys = sorted(keys)
        parameters = collections.OrderedDict([(k, begin[k]) for k in keys])
        parameters.update(ensembleMember=0)
        models = []
        parameters.update(configData.fixedParams())
        parameters.update(refDir=refDirPath)
        self.parameters = []
        for count, dir in enumerate(['zz001', 'zz002']):
            createDir = os.path.join(testDir, dir)
            parameters.update(ensembleMember=count)
            self.parameters.append(parameters.copy())
            models.append(ModelSimulation.ModelSimulation(createDir, name=dir, create=True,
                                                          refDirPath=refDirPath,
                                                          ppExePath=configData.postProcessOutput(),
                                                          ppOutputFile=configData.postProcessOutput(),
                                                          parameters=parameters.copy(), obsNames=configData.obsNames(),
                                                          verbose=False
                                                          ))
            outFile = os.path.join(createDir, configData.postProcessOutput())

            shutil.copy(os.path.join(refDir, '01_GN', 'h0101', 'observables.nc'),
                        outFile)  # copy over a netcdf file of observations.

        mDirs = [m.dirPath for m in models]  # used for testing
        self.modelDirs = mDirs
        self.models = models
        self.config = copy.deepcopy(configData)  # make a COPY of the config.

        self.mSubmit = Submit.ModelSubmit(configData, ModelSimulation.ModelSimulation,
                                          None, config.optFunctions['default'], rootDir=testDir, verbose=True)

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        # self.tmpDir.cleanup() # sadly fails because not all files in are writable.
        optClimLib.delDirContents(self.tmpDir.name)

    def test_init(self):
        """
        Test we worked
        :return:
        """
        # from setup should have a directory with files in. Check!
        for k, m in self.mSubmit._models.items():
            self.assertTrue(os.path.isdir(m.dirPath))
        # should have empty list of models to run.
        self.assertEqual(len(self.mSubmit._modelsToSubmit), 0)
        # should have two models found
        self.assertEqual(len(self.mSubmit._models), 2)
        # config is right
        self.assertEqual(self.config, self.mSubmit.config)
        # verify transform works...
        tMat = self.mSubmit.transMatrix()
        cov = self.config.Covariances(scale=True)['CovTotal']
        got = tMat.dot(cov).dot(tMat.T)
        expect = np.identity(len(self.config.obsNames()))
        nptest.assert_allclose(got, expect, atol=1e-10, err_msg='Transform not giving I')
        self.config.setv('dummy', 2)
        self.assertNotEqual(self.config.Config, self.mSubmit.config.Config)
        # verify that fakeFn works..

    # suspect next set of tests can be done in more pythonic way by iterating over method names or something..
    def test_obsNames(self):
        """Test obsName method words"""

        self.assertEqual(self.config.obsNames(), self.mSubmit.config.obsNames())

    def test_fixParams(self):
        """
        test fixParams method works
        :return:
        """
        expectParams = self.config.fixedParams()
        expectParams.update(refDir=self.mSubmit.refDir)
        self.assertEqual(expectParams, self.mSubmit.fixedParams())

    def test_paramNames(self):
        """
        test fixParams method works
        :return:
        """
        self.assertEqual(self.config.paramNames(), self.mSubmit.paramNames())

    def test_targets(self):
        """
        test target method works
        :return:
        """

        self.assertFalse((self.config.targets() == self.mSubmit.targets()).all())
        self.assertTrue((self.config.targets(scale=True) == self.mSubmit.targets()).all())

    def test_readModelDir(self):
        """
        Test can read modelDir OK!
        :return:
        """
        # create model dir with know params. Read it in.
        # dict should contain 3 (2 from setup +1)
        # and be indexed by key from parameters + fixed Params
        name = 'zz003'
        parameters = self.config.beginParam().to_dict()
        parameters['VF1'] = 4.0
        dir = os.path.join(self.dirPath, name)
        m = ModelSimulation.ModelSimulation(
            dir, name=name, create=True, refDirPath=os.path.join(self.refPath, 'start'),
            ppExePath=self.config.postProcessOutput(), ppOutputFile=self.config.postProcessOutput(),
            parameters=parameters, obsNames=self.config.obsNames(), verbose=False
        )
        # put some fake obs in!
        outFile = os.path.join(m.dirPath, self.config.postProcessOutput())
        shutil.copy(os.path.join(self.refPath, '01_GN', 'h0101', 'observables.nc'),
                    outFile)  # copy over a netcdf file of observations.
        m2 = self.mSubmit.readModelDir(dir)
        self.assertEqual(m.getParams(), m2.getParams())
        self.assertEqual(m, m2)

    def test_genKey(self):
        """
        Tests for genKey
        :return:
        """

        # test key for mixed params is as expected
        pDict = {'zz': 1.02, 'aa': 1, 'nn': [0, 1]}
        expect = ('aa', '1', 'nn', '[0, 1]', 'zz', '1.02')
        key = self.mSubmit.genKey(pDict)
        self.assertEqual(key, expect)
        # test that small real differences don't cause any differences.
        pDict = {'zz': 1.0200001, 'aa': 1, 'nn': [0, 1]}
        key = self.mSubmit.genKey(pDict)
        self.assertEqual(key, expect)

    def test_transMatrix(self):
        """
        tests fpr transMatrix
        :return:
        """

        got = self.mSubmit.transMatrix()
        expect = self.config.transMatrix(scale=True)
        nptest.assert_allclose(got, expect)

        # need to do test that trans matrix behaves as expected..

    def test_modelSubmit(self):
        """
        Tests for modelsSubmit
        :return:
        """

        params = [{'vf1': 2.0, 'rhcrit': 0.9}, {'rhcrit': 0.9, 'vf1': 2.1}]
        for p in params:
            key = self.mSubmit.genKey(p)
            self.mSubmit.modelSubmit((key, p))

        for expect, got in zip(params, self.mSubmit.modelSubmit()):
            self.assertEqual(expect, got)

    def test_model(self):
        """
        Test that model method works
        :return:
        """
        # first case -- one we already have.
        m = self.mSubmit.modelFn(self.modelDirs[0])  # read a model.
        params = m.getParams()  # models contains all params. So need to remove fixed ones.
        # for p in self.mSubmit.fixedParams().keys():
        #    params.pop(p, None)

        # params.update(refDir=m.refDirPath()) # add in refDir. Think this should be done in mSubmit.model?
        print("params are ", params)

        m2 = self.mSubmit.model(params)
        print("model params ", m.getParams())
        print("model2 params", m2.getParams())

        self.assertEqual(m, m2)

    def test_nextName(self):
        """
        Test we can make a generator function and it works.
        :return:
        """
        nameGen = self.mSubmit.nextName()
        name = 'zz003'
        expectDir = self.mSubmit.rootDir / name
        self.assertEqual(next(nameGen), (expectDir, name))
        # add some directories and check get what we expect.
        for d in ['zz005', 'zz010', 'zz012']:
            dir = self.mSubmit.rootDir / d
            dir.mkdir(parents=True)

        dirNo = [4, 6, 7, 8, 9, 11, 13]
        for indx, name in zip(dirNo, nameGen):
            expectName = 'zz0%2.2i' % (indx)
            expectDir = self.mSubmit.rootDir / expectName
            self.assertEqual((expectDir, expectName), name)

        # test maxDIgits logic. First case with two digits and name set to AbC
        self.mSubmit.config.baseRunID('AbC')
        self.mSubmit.config.maxDigits(2)
        nameGen2 = self.mSubmit.nextName()
        dirNo = [1, 2, 3, 4, 5]
        for indx, name in zip(dirNo, nameGen2):
            expectName = 'AbC%2.2i' % (indx)
            expectDir = self.mSubmit.rootDir / expectName
            self.assertEqual((expectDir, expectName), name)

        # test for maxDigits =0
        self.mSubmit.config.baseRunID('AbCde')
        self.mSubmit.config.maxDigits(0)
        nameGen3 = self.mSubmit.nextName()
        result = next(nameGen3)
        expectName = 'AbCde'
        expectDir = self.mSubmit.rootDir / expectName
        self.assertEqual((expectDir, expectName), result)

        with self.assertRaises(ValueError):
            result = next(nameGen3)  # should trigger an error

    def test_submit(self):
        """
        Test submission works.
        :return:
        """

        self.mSubmit.fakeFn = config.easyFake
        submitStat = self.mSubmit.submit(dryRun=True)
        # no new models so expect to have two directories in the filespace and submitStat to be True
        self.assertTrue(submitStat)
        # count dirs
        dirCount = 0
        with os.scandir(self.mSubmit.rootDir) as dirIter:
            for entry in dirIter:
                if entry.is_dir():
                    dirCount += 1

        self.assertEqual(2, dirCount)

        # now add some models
        params = self.mSubmit.config.beginParam().to_dict()
        params.update(ensembleMember=0)
        for value in [2.0, 3.0, 4.0]:
            params['VF1'] = value
            m = self.mSubmit.model(params, update=True)
            self.assertEqual(m, None)  # should get None as dir does not exist.

        # now expect 5 dirs & 3 new ones
        submitStat = self.mSubmit.submit(dryRun=False)

        self.assertEqual(submitStat, (True, 3))
        # count dirs
        dirCount = 0
        with os.scandir(self.mSubmit.rootDir) as dirIter:
            for entry in dirIter:
                if entry.is_dir():
                    dirCount += 1
                    # should look like zzxxx
                    self.assertEqual('zz', entry.name[0:2])
                    # verify have observations file (got through running submitFake)
                    obsFile = self.mSubmit.config.postProcessOutput()
                    path = os.path.join(self.mSubmit.rootDir, entry.name, obsFile)
                    self.assertTrue(os.path.isfile(path), msg='failed to find %s' % path)

        self.assertEqual(5, dirCount)

        # now read in everything and rerun a case. Should have the same no of directories but only one case submitted

    def test_params(self):
        """
        Test params works works
        :return:
        """

        # test all parameters
        params = self.mSubmit.params(includeFixed=True)
        expect = self.parameters
        expect = pd.DataFrame(expect, index=[m.name() for m in self.mSubmit._models.values()])
        self.assertTrue(np.all(expect == params))
        # test behaviour when excluding fixed params
        # remove the fixed parameters
        expect.drop(columns=self.mSubmit.fixedParams(), inplace=True)
        params = self.mSubmit.params(includeFixed=False)
        self.assertTrue(np.all(expect == params))

    def test_obs(self):
        """

        Test obs works.
        :return:
        """

        # need to write some fake ons into model dirs...
        oNames = self.mSubmit.obsNames()
        obs = collections.OrderedDict([(k, v) for (v, k) in enumerate(oNames)])
        expectObs = []
        indx = []
        for count, m in enumerate(self.mSubmit._models.values()):
            obs.update(**{oNames[2]: count})
            expectObs.append(obs.copy())
            m.writeObs(obs)
            indx.append(m.name())
        expectObs = pd.DataFrame(expectObs, index=indx)
        obs = self.mSubmit.obs(scale=False)
        self.assertTrue(np.all(expectObs == obs))
        # and with scaling on.
        obs = self.mSubmit.obs(scale=True)
        scales = self.mSubmit.scales()
        expectObs *= scales
        self.assertTrue(np.all(expectObs == obs))

    def test_rerunModel(self):
        """
        Test that rerunModel works
        :return:
        """
        # will add the same model twice -- so should have only one to rerun.
        self.mSubmit.rerunModel(self.models[0])
        self.mSubmit.rerunModel(self.models[0])
        self.assertEqual(1, len(self.mSubmit._modelsToRerun))
        # run submit and should only have 1 model to run.
        result = self.mSubmit.submit(dryRun=True)
        self.assertEqual(result, (True, 1))

    def test_paramObs(self):
        """
        Test paramObs method.
        :return:
        """

        # store an obs and get it back again.
        params = pd.Series({'VF1': 2.0, 'Experiment': 'COupled', 'EnsembleMember': 2})
        obs = pd.Series({'OLR': 230.3432, 'RSR': 120.332, 'NetFlux': 0.5}).rename('Run1')
        expect = obs.append(params)
        gobs = self.mSubmit.paramObs(params=params, obs=obs)
        self.assertTrue(np.all(gobs == expect))
        # test read works
        gobs = self.mSubmit.paramObs(params=params)
        self.assertTrue(np.all(gobs == expect))
        # make sure that changing obs doesn't cause changes...
        expect.RSR += 1
        self.assertFalse(np.all(gobs == expect))
        # make a bunch of params and obs and test we can get back a dataframe,
        obsL = []
        self.mSubmit.paramObs(clear=True)  # flush it!
        for indx in range(0, 10):
            obs.NetFlux = float(indx)
            obs = obs.rename('run' + str(indx))
            params.VF1 = float(indx)
            obsL.append(obs.append(params).rename(obs.name))
            self.mSubmit.paramObs(params, obs)

        df = pd.DataFrame(obsL)
        got = self.mSubmit.paramObs()  # just get
        self.assertTrue(np.all(got == df))

    def test_optimiseFunction(self):
        """
        Test optimise function works
        :return:
        """

        def fn1(a):
            return a ** 2

        def fn2(a, MODELRUN=None):
            return (a ** 2, MODELRUN)

        def fn3(a, MODELRUN=None, failNan=False, justObs=False):
            return (a ** 2, MODELRUN, failNan, justObs)

        self.mSubmit.optimiseFunction(fn1)
        r = self.mSubmit.optimiseFunction()
        # should fail.
        with self.assertRaises(TypeError):
            got = r(2)

        r = self.mSubmit.optimiseFunction(fn2)  # should work.
        got = r(2)
        self.assertEqual(got, (4, self.mSubmit))

        self.mSubmit.optimiseFunction_args(failNan=True, justObs=False)
        r = self.mSubmit.optimiseFunction(fn3)  # should work.
        got = r(2)
        self.assertEqual(got, (4, self.mSubmit, False, False))

    def test_runOptimized(self):
        """

        test runOptimized!

        Tests:
            Use idiot test function -- which should not do anything clever.
            And generate configuration too.
            config as nEns not present -- runs one case
            nEns set to one -- runs one case
            nEns set to 2 -- runs two cases.

            Change configuration name -- should  run fresh cases.

        """

        # case 1 -- want to not set nensemble but do modify baseRunId and maxDigits
        config = self.mSubmit.config  # reduce typing -- note this is (as normal in python) a pointer to the config!
        config.baseRunID('test1')
        config.maxDigits(0)
        params = dict(CT=1e-4, EACF=0.5, ENTCOEF=3, ICE_SIZE=3e-5, RHCRIT=0.7, VF1=0.5, CW_LAND=2e-4)
        config.optimumParams(
            **params)  # really should be passed as a series. TODO: Fix all code to use pd.Series undernearth!
        # runOptimized should raise runModelError as needs to run new things.
        with self.assertRaises(Submit.runModelError):
            self.mSubmit.runOptimized()
        # and check what is to be created is as expected.
        nmodels = len(self.mSubmit.modelSubmit())
        expect = 1
        self.assertEqual(nmodels, expect, msg=f'Expected {expect} got {nmodels}')

        # increase ensemble to four and shorten basename.
        config.baseRunID('test')
        config.maxDigits(1)
        config.ensembleSize(4)
        with self.assertRaises(Submit.runModelError):
            self.mSubmit.runOptimized()
        nmodels = len(self.mSubmit.modelSubmit())
        expect = 4
        self.assertEqual(nmodels, expect, msg=f'Expected {expect} got {nmodels}')


# TODO add a test case for optimiseFn().

if __name__ == "__main__":
    print("Running Test Cases")
    unittest.main()  # actually run the test cases
