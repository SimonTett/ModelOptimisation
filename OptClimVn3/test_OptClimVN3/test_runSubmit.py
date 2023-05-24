"""
Place to put tests for Submit.
"""

import copy
import os
import shutil
import tempfile
import typing
import unittest
import unittest.mock
import pandas as pd
import pathlib  # make working with file paths easier. TODO make universal.
import functools  # want the partial!
import numpy as np
import numpy.testing as nptest
import exceptions
import runSubmit
import SubmitStudy
import StudyConfig
import genericLib
from Model import Model
from HadCM3 import HadCM3


def fake_fn(params: dict,
            config: typing.Optional[StudyConfig.OptClimConfigVn2] = None,
            var_scales: typing.Optional[pd.Series] = None,
            rng=None):
    """
    Wee test fn for trying out things.
    :param params -- dict of parameter values
    :param config -- configuration -- default is None. If not available then fn will crash.
    :param var_scales -- scales to apply to variables.
        Should be a pandas series. If not provided no scaling will be done
    :param rng -- random number generator. Default None
      If provided than random multivariate noise  based on the internal variance covariance will be added.

    for everything but params given design of optimisation algorithms you will need to find a way of getting
      the extra params in. One way is to make a lambda fn. Another is to wrap it in a function.

    returns  "fake" data as a pandas Series
    """
    import logging
    logging.debug("faking with params: " + str(params))
    tgt = config.targets()
    pranges = config.paramRanges()
    min_p = pranges.loc['minParam', :]
    max_p = pranges.loc['maxParam', :]
    scale_params = max_p - min_p
    pscale = (pd.Series(params) - min_p) / scale_params
    pscale -= 0.5  # tgt is at params = 0.5
    result = 10 * (pscale + pscale ** 2)
    # this fn has one minima and  no maxima between the boundaries and the minima. So should be easy to optimise.
    delta_len = len(tgt) - result.shape[-1]
    if delta_len > 0:
        result = np.append(result, np.zeros(delta_len), axis=-1)  # increase result
    result = pd.Series(result.values[0:len(tgt)], index=tgt.index)  # brutal conversion to obs space.
    if var_scales is not None:
        result /= var_scales  # make sure changes are roughly comparable size after scaling.

    result += tgt
    if rng is not None:
        intVar = config.Covariances()['CovIntVar']
        result += rng.multivariate_normal(tgt.values, intVar)  # add in some noise usually not needed for testing

    return result


class testRunSubmit(unittest.TestCase):
    """
    Test cases for runSubmit. There should be one for every method in runSubmit.

    """

    def setUp(self):
        """
        Standard setup for all test cases
        :return:
        """

        self.tmpDir = tempfile.TemporaryDirectory()
        cpth = runSubmit.runSubmit.expand("$OPTCLIMTOP/OptClimVn3/configurations/dfols14param_opt3.json")
        config = StudyConfig.readConfig(cpth)
        self.tmpDir = tempfile.TemporaryDirectory()
        testDir = pathlib.Path(self.tmpDir.name)
        refDir = runSubmit.runSubmit.expand("$OPTCLIMTOP/Configurations/xnmea")

        self.refDir = refDir  # where the default reference configuration lives.
        self.rootDir = testDir  # where

        # generate a runSubmit object with some data in it.
        #self.rSubmit = SubmitStudy.SubmitStudy(config, name='test_study', rootDir=self.rootDir,
        #                                       model_name='HadCM3', computer='SGE', refDir=self.refDir)
        self.config = copy.deepcopy(config)  # make a COPY of the config.

    def tearDown(self):
        """
        Clean up by removing the temp directory contents
        :return:
        """
        # self.tmpDir.cleanup() # sadly fails because not all files in are writable.
        # optClimLib.delDirContents(self.tmpDir.name)
        self.tmpDir.cleanup()
        # shutil.rmtree(self.tmpDir)

    # test case for _stdFunction

    def test_stdFunction(self):

        """
        test stdFunction

        Tests:

        0) Can retrieve existing values and they are as expected.

        1)
            run case with ensembleSize = 1 and 1 param vector set. Should get vector of Nan.
            Should be 1 model to run.

        2)  run case with ensembleSize = 2 and 1 param vector set. Should get vector of Nan.
            Should be 2 models to run. One case is the same as test #1

        3)  run case with ensembleSize = 2 and 2 param vector set. Should get 2 vectors of Nan.
            Should be 6 models to run. The four models generated here + 2 models from case 2
        """
        # setup
        configData = self.config
        begin = configData.beginParam()
        rSubmit = self.rSubmit  # setup runSubmit for ease of use.
        nparam = len(configData.paramNames())
        nobs = len(configData.obsNames())
        params = np.repeat(1, nparam)

        params2 = np.vstack([params, np.repeat(0, nparam)])
        # test 0 -- find existing case.

        p = rSubmit.params().drop(columns='ensembleMember')
        p = p.iloc[0, :]  # parameters we want.
        nobs = len(rSubmit.obsNames())
        data = rSubmit.stdFunction(p.values)
        expect = rSubmit.obs(scale=False).iloc[0, :]
        nptest.assert_allclose(data, expect.values)

        # test 1
        rSubmit.config.ensembleSize(1)  # 1 member ensemble
        with self.assertRaises(exceptions.runModelError) as context:
            result = rSubmit.stdFunction(params)  # should get runModelError exception.

        # no of models to run should be 1.
        models = rSubmit.modelSubmit()
        self.assertEqual(len(models), 1, "Expect only 1 model to submit")
        # and then test with out exception
        result = rSubmit.stdFunction(params, raiseError=False)
        # only 1 model  so should be a nObs element vector
        self.assertEqual(result.shape, (nobs,), 'Expected only 1 result')
        self.assertTrue(np.all(np.isnan(result)), 'ALl values should be Nan')
        # no of models to run should be 1.
        models = rSubmit.modelSubmit()
        self.assertEqual(len(models), 1, "Expect only 1 model to submit")

        # test 2. Set ensemble size to 2.
        rSubmit.config.ensembleSize(2)  # 1 member ensemble
        with self.assertRaises(exceptions.runModelError) as context:
            result = rSubmit.stdFunction(params)

        # no of models to run should be 2
        models = rSubmit.modelSubmit()
        self.assertEqual(len(models), 2, "Expect only 2 model to submit")

        result = rSubmit.stdFunction(params, raiseError=False)
        # only 1 result so should be a nobs vector
        self.assertEqual(result.shape, (nobs,), 'Expected one element vector')
        self.assertTrue(np.all(np.isnan(result)), 'ALl values should be Nan')
        # no of models to run should be 1.
        models = rSubmit.modelSubmit()
        self.assertEqual(len(models), 2, "Expect 2 models to submit")

        # test 3. Have multiple params

        rSubmit.config.ensembleSize(2)  # 2 member ensemble
        with self.assertRaises(exceptions.runModelError) as context:
            result = rSubmit.stdFunction(params2)

        # no of models to run should be 4.. (ensemble member * param2)
        models = rSubmit.modelSubmit()
        nmodel = len(models)
        self.assertEqual(nmodel, 4, f"Expect 4 models to submit. Got {nmodel}")
        # test without raising exception.
        result = rSubmit.stdFunction(params2, raiseError=False)
        # 2 results so should be a 2 element vector
        self.assertEqual(result.shape[0], 2, 'Expected two element vector')
        # everything should be nan
        self.assertTrue(np.all(np.isnan(result)), 'Expected all to be missing')
        self.assertEqual(result.shape, (2, nobs), 'size not as expected')

        # no of models to run should be 4.. (ensemble member * param2)
        models = rSubmit.modelSubmit()
        nmodel = len(models)
        self.assertEqual(nmodel, 4, f"Expect 4 models to submit. Got {nmodel}")

        # check if we run with df=True we get a dataframe...
        p = rSubmit.params().drop(columns='ensembleMember').iloc[0, :].values  # cases that have already been run
        result = rSubmit.stdFunction(p, df=True)  # should get back a 1 row dataframe.
        self.assertTrue(isinstance(result, pd.DataFrame), 'Should be dataframe')
        self.assertEqual(result.shape, (1, nobs), 'Size not expected')

        # check if we apply a transform that works. Only verify size and column names not values..
        trans = rSubmit.transMatrix(dataFrame=True)
        # will truncate this...
        trans = trans.iloc[0:5, :]
        # using params from prev test
        result = rSubmit.stdFunction(p, df=True, transform=trans)
        # check that result is 1x5 df with columns being 0..4
        expect_cols = np.arange(0, 5)
        nptest.assert_equal(expect_cols, result.columns)
        self.assertEqual(result.shape, (1, 5))

        # test if residual works.
        expect = rSubmit.stdFunction(p, df=True, scale=True) - rSubmit.targets(scale=True)
        result = rSubmit.stdFunction(p, df=True, scale=True, residual=True)
        self.assertTrue(expect.equals(result), 'Residual test failed')

        # test if sum of squares work
        # just use the previous test results...
        expect = (expect ** 2).sum(axis=1)
        result = rSubmit.stdFunction(p, df=True, scale=True, residual=True, sumSquare=True)
        self.assertTrue(expect.equals(result), 'SumSquare test failed')
        # and we are a series of size 1.
        self.assertEqual(result.size, 1, 'Size not as expected')

    # test case for stdFunction

    def test_genOptFunction(self):
        """
        Test stdFunction which should return a function ready to drop into optimise/etc
         Will test by just running the fn and seeing we get a result!

        """

        # lets run  it
        configData = self.config
        rSubmit = self.rSubmit  # setup runSubmit for ease of use.
        configData.ensembleSize(2)

        params = rSubmit.params().drop(columns='ensembleMember')
        params = params.iloc[0, :].values
        nobs = len(rSubmit.obsNames())
        # test 1 -- result should be a vector of len nobs
        fn = rSubmit.genOptFunction(scale=True)
        result = fn(params)
        self.assertEqual(result.shape, (nobs,), f'Expected {nobs} element array')
        expect = rSubmit.obs(scale=True).mean()
        nptest.assert_allclose(expect, result)

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

            pass in an optimum config. Should run with those values.
        """
        configData = self.config
        rSubmit = runSubmit.runSubmit(configData, self.Model,
                                      None, rootDir=self.testDir, verbose=self.verbose, fakeFn=self.fake_fn)
        # case 1 -- want to not set nensemble but do modify baseRunId and maxDigits

        configData.baseRunID('test1')
        configData.maxDigits(0)
        params = dict(CT=1e-4, EACF=0.5, ENTCOEF=3, ICE_SIZE=3e-5, RHCRIT=0.7, VF1=0.5, CW=2e-4)
        configData.optimumParams(
            **params)  # really should be passed as a series. TODO: Fix all code to use pd.Series underneath!
        # runOptimized should raise runModelError as needs to run new things.
        with self.assertRaises(exceptions.runModelError):
            rSubmit.runOptimized()
        # and check what is to be created is as expected.
        nmodels = len(rSubmit.modelSubmit())
        expect = 1
        self.assertEqual(nmodels, expect, msg=f'Expected {expect} got {nmodels}')

        # increase ensemble to four and shorten basename.
        configData.baseRunID('test')
        configData.maxDigits(1)
        configData.ensembleSize(4)
        with self.assertRaises(exceptions.runModelError):
            rSubmit.runOptimized()
        nmodels = len(rSubmit.modelSubmit())
        expect = 4
        self.assertEqual(nmodels, expect, msg=f'Expected {expect} got {nmodels}')

        optConfig = copy.deepcopy(configData)
        opt = optConfig.optimumParams()
        opt *= 1.1  # multiply everything by 1.1
        optConfig.optimumParams(**opt.to_dict())  # TODO change optimumParams to get a series.
        with self.assertRaises(exceptions.runModelError):
            rSubmit.runOptimized(optConfig=optConfig)
        nmodels = len(rSubmit.modelSubmit())  # should generate another 4 runs to do. Making 8.
        expect = 8
        self.assertEqual(nmodels, expect, msg=f'Expected {expect} got {nmodels}')
        # let's run the fake stuff and then a final run of runOptimized.

        status, nModels, finalConfig = rSubmit.submit(restartCmd=None, verbose=self.verbose, cost=True,
                                                      scale=False)
        # need to read the dirs.
        rSubmit = runSubmit.runSubmit(configData, self.Model,
                                      None, rootDir=self.testDir, verbose=self.verbose, fakeFn=self.fake_fn)
        endConfig = rSubmit.runOptimized(optConfig=optConfig)

    def test_runDFOLS(self):
        """
        test runDFOLS

        Note that "noise" is tricky and you need to be careful...
        The noise scaling matters. Can estimate it from the covariance of internal
        var in a climate model. Test case has no noise (though that depends on fake_fn defined in setup for test.
        If wanted note it is tricky and needs care!

        """
        import dfols
        import pandas as pd
        import numpy as np

        from numpy.random import default_rng
        scale = True  # applying scaling or not. Need to apply consistently

        configData = copy.deepcopy(self.config)
        configData.postProcessOutput('obs.nc')  # make output netcdf to give increased precision though I think some
        # precision is lost.

        # Set begin to max value.
        maxP = configData.paramRanges().loc['maxParam', :]
        begin = configData.beginParam(maxP)  # can truncate here using .iloc[0:x] if wanted.

        obs = configData.obsNames()
        nobs = len(begin)
        obs = obs[0:nobs]
        scales = configData.scales()
        configData.scales(scales[0:nobs])
        configData.obsNames(obs, add_constraint=False)
        # truncate the obsNames. (does not have to be the same as params but makes life easier if so)
        configData.constraint(False)  # no constraint
        obs = configData.obsNames()
        varParamNames = configData.paramNames()  # extract the parameter names if have them
        covTotal = pd.DataFrame(np.identity(nobs), index=obs, columns=obs) * 1e-5  # small random error..
        # Work out scaling on variables from the scales in the config. (so when scaling on "interesting" things happen)
        if scale:
            var_scales = 10.0 ** np.round(np.log10(configData.scales()))
        else:  # no scaling so want values to be comparable.
            var_scales = pd.Series(1.0 + np.arange(nobs) / 10., index=obs)
        covTotal = covTotal / (var_scales ** 2)  # taking advantage of diagonal covariance.
        configData.Covariances(CovTotal=covTotal)  # set total covar.
        tgt = configData.targets(scale=False)
        if scale:
            scales = configData.scales()
        else:
            scales = pd.Series(np.ones(nobs), index=obs)

        # need to modify the function a bit. Apply scaling and Transform.
        # This makes it compatible with way that DFOLS gets run with framework.
        Tmat = configData.transMatrix(scale=scale)

        def fn_opt(params):
            pDict = {pname:pvalue for pname,pvalue in zip(configData.paramNames(),params)}
            result = fake_fn(pDict, config=configData, var_scales=var_scales)
            result -= tgt  # remove tgt
            result *= scales  # scale
            result = result @ Tmat.T.values  # apply transformation.
            return result

        minP = configData.paramRanges().loc['minParam', :].values
        maxP = configData.paramRanges().loc['maxParam', :].values
        rangeP = maxP - minP

        expect = np.repeat(0.5, nobs) * rangeP + minP  # expected values
        nptest.assert_allclose(fn_opt(expect), 0, atol=1e-4)
        # for expected params should get back array close to 0
        # run DFOLS "naked"
        dfols_config = configData.DFOLS_config()
        dfols_config['maxfun'] = 50
        dfols_config['rhobeg'] = 1e-1
        dfols_config['rhoend'] = 1e-3
        # general configuration of DFOLS -- which can be overwritten by config file
        userParams = {'logging.save_diagnostic_info': True,
                      'logging.save_xk': True,
                      'noise.additive_noise_level': nobs * 1e-4,  # upper est of noise.
                      'general.check_objfun_for_overflow': False,
                      'init.run_in_parallel': False,
                      'interpolation.throw_error_on_nans': True,  # make an error happen!
                      }
        overwrite = {'noise.quit_on_noise_level': None,
                     'noise.additive_noise_level': nobs * 1e-4,
                     }

        # update the user parameters from the configuration.
        userParams = configData.DFOLS_userParams(userParams=userParams)
        userParams.update(overwrite)
        # and update the config...
        configData.DFOLS_config(dfols_config)
        configData.DFOLS_userParams(updateParams=userParams)
        np.random.seed(123456)  # make sure RNG is initialised to same value for first eval of dfols.
        # This should be the same seed as used in runDFOLS
        prange = (configData.paramRanges(paramNames=varParamNames).loc['minParam', :].values,
                  configData.paramRanges(paramNames=varParamNames).loc['maxParam', :].values)
        solution = dfols.solve(fn_opt, begin.values,
                               objfun_has_noise=True,
                               bounds=prange, scaling_within_bounds=True
                               , maxfun=dfols_config.get('maxfun', 100)
                               , rhobeg=dfols_config.get('rhobeg', 1e-1)
                               , rhoend=dfols_config.get('rhoend', 1e-3)
                               , print_progress=False
                               , user_params=userParams
                               )

        if solution.flag not in (solution.EXIT_SUCCESS, solution.EXIT_MAXFUN_WARNING):
            print("dfols failed with flag %i error : %s" % (solution.flag, solution.msg))
            raise Exception("Problem with dfols")
        print(f"DFOLS finished {solution.flag} {solution.msg}")
        soln = pd.Series(solution.x, index=varParamNames).rename(f'Naked DFOLS')
        expect = pd.Series(expect, index=varParamNames).rename('expect')
        df = pd.DataFrame([expect, soln])
        #    expect to be within 0.01% of the expected soln.
        nptest.assert_allclose(soln.values, expect.values, rtol=1e-4)

        # now lets go and try runDFOLS
        fake_function = functools.partial(fake_fn, config=configData, var_scales=var_scales)
        fake_function.__name__=f'Faking using {fake_fn.__name__}'
        # set up fake fn to avoid needing a Q system
        run = True
        iterCount = 0
        rSubmit = runSubmit.runSubmit(configData, 'DFOLS_test',
                                          rootDir=self.rootDir, refDir=self.refDir)
        # setup Submit object.
        while run:

            try:
                finalConfig = rSubmit.runDFOLS(scale=scale)  # run DFOLS
                run = False  # if we get to here then we are done.
            except exceptions.runModelError:  # Need to run some models.
                rSubmit.instantiate()  # instantiate all models that need running.
                status = rSubmit.submit_all_models(fake_fn=fake_function)
                self.assertTrue(status)
                iterCount += 1
                # expect nobs+1 on first iteration (parallel running)
                if iterCount == 1:
                    self.assertEqual(nobs + 1, len(rSubmit.model_index),
                                     f'Expected to have {nobs + 1} models ran on iteration#1')

        best = finalConfig.optimumParams()
        df = df.append(best.rename('DFOLS'))
        transJac = pd.DataFrame(solution.jacobian, index=varParamNames, columns=Tmat.index)
        finalConfig.transJacobian(transJac)  # set the transformed Jacobian

        print("All done with result\n", df)
        info, transJac = finalConfig.get_dataFrameInfo(['diagnostic_info', 'transJacobian'])
        print("Info\n", info)
        nptest.assert_allclose(finalConfig.get_dataFrameInfo('transJacobian', dtype=float), solution.jacobian,
                               atol=1e-10)  # check Jacobian as stored is right
        #    expect to be within 0.01% of the expected soln. If random covariance done then this will be a
        # lot bigger
        nptest.assert_allclose(best.values, expect.values, rtol=1e-4)

    def test_runJacobian(self):
        """
        test runJacobian!

        Drive with extreme cases for all  parameters should get dp towards centre (so easy to compute)
        Check jacobian from bare run and compare with runJacobian. Expect two goes at runJacobian and one submit.

        Test that optConfig works.
        """
        # Make the optimum values be max values (so we know perturbations take us to the center)

        # Modify the steps to be small but different.
        configData = self.config
        steps = configData.steps()
        steps.iloc[:] = np.arange(len(steps)) * 0.001 + 0.1  # want small but not equal steps.
        steps.loc['scale_steps'] = True
        configData.steps(steps=steps)  # set it in the configuration
        steps = configData.steps()  # get the steps we are actually using.
        configData.constraint(False)  # turn of constraint.
        prange = configData.paramRanges()  # param ranges
        # need number of obs and params to be the same -- will set to 7.
        optParams = prange.loc['maxParam'].iloc[0:7]
        paramNames = optParams.index
        configData.paramNames(paramNames=paramNames)  # set the param names to reduced set.
        obsNames = configData.obsNames(add_constraint=False)  # get the obs names making sure constraint not used.
        configData.obsNames(obsNames=obsNames[0:7])  # shorten obsnames.
        obsNames = configData.obsNames(add_constraint=False)  # get the obs names making sure constraint not used.
        configData.optimumParams(**optParams.to_dict())  # these are scaled values.
        var_scales = optParams.copy()
        var_scales.iloc[:] = 1
        fake_fn = functools.partial(config.fake_fn, var_scales=var_scales)
        fn = functools.partial(config.bare_fn, var_scales=var_scales, config=configData)
        fn2 = lambda x: pd.Series(fn(x), index=obsNames)
        # all wrapped so we get a Series
        nparam = len(optParams)
        # compute what we expect -- BROKEN as need to set the index correctly...
        ref = fn2(optParams.values)
        jac_bare = []
        for p in steps.index:
            dd = optParams.copy()
            dd.loc[p] -= steps.loc[p]
            delta = fn2(dd.values) - ref
            delta = delta.rename(p)  # name it by param
            jac_bare.append(delta)
        jac_bare = pd.DataFrame(jac_bare)  # convert to dataframe
        # and convert the estimate of the Jacobian
        jac_bare = jac_bare.div(-steps, axis=0)
        # and apply linear transform
        Tmat = configData.transMatrix()
        jac_bare = jac_bare @ Tmat.T
        # print("raw jacobian: \n",jac_bare)

        rSubmit = runSubmit.runSubmit(configData, self.Model,
                                      None, rootDir=self.testDir, fakeFn=fake_fn, verbose=self.verbose)
        # setup Submit object.
        try:
            finalConfig = rSubmit.runJacobian()  # run Jacobian

        except exceptions.runModelError:  # Need to run some models.
            status, nModels, finalConfig = rSubmit.submit(restartCmd=None, verbose=self.verbose)
            # expect nparam+1 models
            self.assertEqual(nparam + 1, nModels, f'Expected to have {nparam + 1} models ran on iteration#1')

        # and read all data back in..
        rSubmit = runSubmit.runSubmit(configData, self.Model,
                                      None, rootDir=self.testDir, fakeFn=fake_fn, verbose=self.verbose)
        finalConfig = rSubmit.runJacobian()  # run Jacobian to get the final answer
        # now check jac is what we expect...
        jac_run = finalConfig.transJacobian()
        nptest.assert_allclose(jac_run, jac_bare, atol=1e-9)  # round trip through json removes some precision.

        # check that passing in optConfig works and that have nparam +1 cases.
        optConfig = copy.deepcopy(self.config)
        opt = optConfig.optimumParams()
        opt *= 0.9
        optConfig.optimumParams(**opt.to_dict())
        done = False
        while done:
            try:
                finalConfig = rSubmit.runJacobian(optConfig)  # run Jacobian
                done = True
            except exceptions.runModelError:  # Need to run some models.
                status, nModels, finalConfig = rSubmit.submit(restartCmd=None, verbose=self.verbose)
                # expect nparam+1 models
                self.assertEqual(nparam + 1, nModels, f'Expected to have {nparam + 1} models ran on iteration#1')
        # now check jac is what we expect...
        jac_run = finalConfig.transJacobian()
        nptest.assert_allclose(jac_run, jac_bare, atol=1e-9)

    def test_runGaussNewton(self):
        """

        Test Gauss Newton algorithm.
        Tests are:
        1) Get expected result from (modified) "bare_fn"
        2) Get same result from fake_fn & pattern is nparam runs (computing Jacobian/Hessian) +  nalphas (linesearch cpt)
        More complex tests should be done with the module itself. This just tests it works in the context of generating new runs.
        """
        import Optimise

        scale = True  # applying scaling or not. Need to apply consistently

        configData = self.config
        configData.postProcessOutput('obs.json')  # make output netcdf to give increased precision though I think some
        # precision is lost.

        # Set beginning values to max
        begin = configData.beginParam(configData.paramRanges().loc['maxParam', :])  # and also resets param names

        obs = configData.obsNames()
        nobs = len(begin)
        obs = obs[0:nobs]
        scales = configData.scales()
        configData.scales(scales[0:nobs])
        configData.obsNames(obs, add_constraint=False)
        # truncate the obsNames. (does not have to be the same as params but makes life easier if so)
        configData.constraint(False)  # no constraint
        obs = configData.obsNames()

        # code below is a lift from test_DFOLS. TODO put the common code in the setup.

        if scale:
            var_scales = 10.0 ** np.round(np.log10(configData.scales()))
        else:  # no scaling so want values to be comparable.
            var_scales = pd.Series(1.0 + np.arange(nobs) / 10., index=obs)

        covTotal = pd.DataFrame(np.identity(nobs), index=obs, columns=obs) * 1e-5  # small random error..
        covTotal = covTotal / (var_scales ** 2)  # taking advantage of diagonal covariance.
        configData.Covariances(CovTotal=covTotal, CovIntVar=0.1 * covTotal,
                               CovObsErr=0.9 * covTotal)  # set total covar.
        tgt = configData.targets(scale=False)
        if scale:
            scales = configData.scales().values
        else:
            scales = np.ones(nobs)

        # need to modify the function a bit. Apply scaling and Transform.
        # This makes it compatible with way that DFOLS gets run with framework.
        Tmat = configData.transMatrix(scale=scale)

        def fn_opt(params):
            result = config.bare_fn(params, config=configData, var_scales=var_scales)
            print(len(result), len(tgt))
            result -= tgt.values
            result *= scales  # scale
            result = result @ Tmat.T.values  # apply transformation.
            return result

        minP = configData.paramRanges().loc['minParam', :].values
        maxP = configData.paramRanges().loc['maxParam', :].values
        rangeP = maxP - minP

        expect = np.repeat(0.5, nobs) * rangeP + minP  # expected values
        nptest.assert_allclose(fn_opt(expect), 0.0, atol=1e-4)

        # run bare optimise.
        optimise = configData.optimise().copy()  # get optimisation info
        intCov = configData.Covariances(scale=scale)['CovIntVar']
        # Scaling done for compatibility with optFunction.
        # need to transform intCov. errCov should be I after transform.

        intCov = Tmat.dot(intCov).dot(Tmat.T)
        # This is correct-- it is the internal covariance transformed
        optimise['sigma'] = False  # wrapped optimisation into cost function.
        optimise['deterministicPerturb'] = True  # deterministic perturbations.
        optimise['maxIterations'] = 10  # no more than 10 iterations.
        paramNames = configData.paramNames()
        nObs = Tmat.shape[0]  # could be less than the "raw" obs depending on Tmat.
        start = configData.beginParam()
        print("start", len(start))
        best, status, info = Optimise.gaussNewton(fn_opt, start.values,
                                                  configData.paramRanges(paramNames=paramNames).values.T,
                                                  configData.steps(paramNames=paramNames).values,
                                                  np.zeros(nObs), optimise,
                                                  cov=np.identity(nObs), cov_iv=intCov, trace=False)

        # expect to have converged and that best is close to expect
        self.assertEqual(status, 'Converged', msg='Expected to have converged')
        nptest.assert_allclose(best, expect, atol=1e-3)  # close to 0.1%. Can probably do better by hacking a bit..
        # get out the jacobian for later comparision.
        jac = info['jacobian'][-1, :, :]  # bare right now...
        print("=" * 60)
        # now to run runGaussNewton.
        # first clean up testDir by deleteting everything in it.
        optClimLib.delDirContents(self.testDir)
        run = True
        iterCount = 0
        nalpha = len(optimise['alphas'])
        nparam = len(expect)
        fake_fn = functools.partial(config.fake_fn, var_scales=var_scales)
        # set up fake fn to avoid needing a Q system

        # Having trouble with fake_fn/stdFunction and fn_opt. So need to test they give same results
        # for param choice...

        # test that opt_fn (as ran by runGaussNewton and fn_opt give the same results.
        # a bit of a pain as need to set up the runSubmit
        rSubmit = runSubmit.runSubmit(configData, self.Model,
                                      None, rootDir=self.testDir, fakeFn=fake_fn, verbose=self.verbose)
        rSubmit.stdFunction(begin.values, df=True, raiseError=False, transform=Tmat, scale=scale,
                            residual=True)  # say we want it.
        status, nModels, finalConfig = rSubmit.submit(restartCmd=None,
                                                      verbose=self.verbose)  # run it (which generates it)
        rSubmit = runSubmit.runSubmit(configData, self.Model,
                                      None, rootDir=self.testDir, fakeFn=fake_fn,
                                      verbose=self.verbose)  # read it back in (in effect)
        got = rSubmit.stdFunction(begin.values, df=True, raiseError=False, transform=Tmat, scale=scale,
                                  residual=True, verbose=True)
        got_fn_opt = pd.Series(fn_opt(begin.values), index=Tmat.index)

        nptest.assert_allclose(got_fn_opt, got.iloc[0, :])
        # first clean up testDir by deleteting everything in it.
        optClimLib.delDirContents(self.testDir)
        while True:
            rSubmit = runSubmit.runSubmit(configData, self.Model,
                                          None, rootDir=self.testDir, fakeFn=fake_fn, verbose=self.verbose)
            try:
                finalConfig = rSubmit.runGaussNewton(verbose=self.verbose, scale=scale)
                break  # done with running
            except exceptions.runModelError:
                status, nModels, finalConfig = rSubmit.submit(restartCmd=None, verbose=self.verbose, cost=True,
                                                              scale=scale)

                iterCount += 1
                # expect nparam+1 on first iteration (parallel running)
                if iterCount == 1:
                    self.assertEqual(nparam + 1, nModels, f'Expected to have {nobs + 1} models ran on iteration#1')
                elif (iterCount % 2) == 0:
                    self.assertEqual(nalpha, nModels, f'Expected to have {nalpha} models ran on iteration# {iterCount}')
                else:
                    self.assertEqual(nparam, nModels, f'Expected to have {nparam} models ran on iteration# {iterCount}')

        # expect optimum value to be close to expected value.
        best = finalConfig.optimumParams()
        expect = pd.Series(expect, index=paramNames).rename('expect')
        print(pd.DataFrame([best, expect]))
        nptest.assert_allclose(best, expect, rtol=5e-4)
        nptest.assert_allclose(finalConfig.get_dataFrameInfo('transJacobian', dtype=float), jac,
                               atol=1e-10)  # check Jacobian as stored is right

        # check that setting maxIterations to 1 only has 1 iteration.
        optClimLib.delDirContents(self.testDir)
        configData.optimise(maxIterations=1)  # limit to 1 iteration
        while True:
            rSubmit = runSubmit.runSubmit(configData, self.Model,
                                          None, rootDir=self.testDir, fakeFn=fake_fn, verbose=self.verbose)
            try:
                finalConfig = rSubmit.runGaussNewton(verbose=self.verbose, scale=scale)
                break  # done with running
            except exceptions.runModelError:
                status, nModels, finalConfig = rSubmit.submit(restartCmd=None, verbose=self.verbose, cost=True,
                                                              scale=scale)
                print("submitting ", status, nModels)

        self.assertEqual(finalConfig.GNparams().Iteration.size, 1, msg=f"Expected 1 iteration got {iterCount}")


if __name__ == "__main__":
    print("Running Test Cases")
    unittest.main()  # actually run the test cases
