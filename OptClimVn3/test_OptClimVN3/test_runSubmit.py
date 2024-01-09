"""
Place to put tests for Submit.
"""

import copy
import pathlib  # make working with file paths easier.
import shutil
import tempfile
import typing
import unittest
import unittest.mock
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pandas.testing as pdtest

import StudyConfig
import optclim_exceptions
import runSubmit
from genericLib import fake_fn

from Models import *

def fake_run(rSubmit: runSubmit, scale: bool = True) -> typing.Callable:
    """ Instantiate and  run fake fns.
    :return the function used to fake.
    """
    config = rSubmit.config
    fake_function = lambda pDict: fake_fn(config,pDict)

    rSubmit.instantiate()
    rSubmit.submit_all_models(fake_fn=fake_function)
    return fake_function

import engine

class testRunSubmit(unittest.TestCase):
    """
    Test cases for runSubmit. There should be one for every method in runSubmit.

    """


    def setUp(self):
        """
        Standard setup for all test cases
        :return:
        """

        tmpDir = tempfile.TemporaryDirectory()
        cpth = runSubmit.runSubmit.expand("$OPTCLIMTOP/OptClimVn3/configurations/dfols14param_opt3.json")
        config = StudyConfig.readConfig(cpth)
        # "fix" config in various ways
        config.postProcessOutput('obs.nc')  # make output netcdf to give increased precision though I think some
        # precision is lost.
        config.constraint(False)  # no constraint
        obs = config.obsNames()
        var_scales = 10.0 ** np.round(np.log10(config.scales()))
        covTotal = pd.DataFrame(np.diag(1.0/var_scales**2), index=obs, columns=obs) *1e-5  # small random error..
        config.Covariances(CovTotal=covTotal)  # set total covar.
        rootDir = pathlib.Path(tmpDir.name)
        refDir = runSubmit.runSubmit.expand("$OPTCLIMTOP/Configurations/xnmea")
        self.rSubmit = runSubmit.runSubmit(config, 'test',
                            rootDir=rootDir, refDir=refDir)

        self.refDir = refDir  # where the default reference configuration lives.
        self.rootDir = rootDir  # where the config info will go.
        self.tmpDir = tmpDir
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

        4) Exceed max_model_simulations. Show raise an error.
        """
        # setup
        configData = copy.deepcopy(self.config)
        begin = configData.beginParam()
        rSubmit = self.rSubmit  # setup runSubmit for ease of use.
        nparam = len(configData.paramNames())
        nobs = len(configData.obsNames())
        params = np.ones(nparam)
        params2 = np.vstack([params, np.repeat(0, nparam)])
        # test 0 -- run 1 model

        data = rSubmit.stdFunction(params,raiseError=False) # do not raise error.
        expect = np.repeat(np.nan,nobs)
        nptest.assert_equal(data,expect)
        # no of models to run should be 1.
        models = rSubmit.model_index.values()
        self.assertEqual(len(models), 1, "Expect only 1 model to submit")

        # test 1 # run same model and should raise ValueError as status is not processed
        rSubmit.config.ensembleSize(1)  # 1 member ensemble
        with self.assertRaises(ValueError): # should also get a warning.
            result = rSubmit.stdFunction(params)  # should get ValueError as running twice.
        # no of models to run should be 1. (as we already have it just asked for it twice)
        models = rSubmit.model_index.values()
        self.assertEqual(len(models), 1, "Expect only 1 model to submit")
        rSubmit.delete() # restart.

        # test 2. Set ensemble size to 2.
        rSubmit.config.ensembleSize(2)  # 1 member ensemble
        with self.assertRaises(optclim_exceptions.runModelError):
            result = rSubmit.stdFunction(params)

        # no of models to run should be 2
        models = rSubmit.model_index.values()
        self.assertEqual(len(models), 2, "Expect only 2 model to submit")
        rSubmit.delete()

        # test 3. Have multiple params

        rSubmit.config.ensembleSize(2)  # 2 member ensemble
        with self.assertRaises(optclim_exceptions.runModelError):
            result = rSubmit.stdFunction(params2)

        # no of models to run should be 4.. (ensemble member * param2)
        models = rSubmit.model_index.values()
        nmodel = len(models)
        self.assertEqual(nmodel, 4, f"Expect 4 models to submit. Got {nmodel}")
        # test without raising exception by setting status to processed and including simulated_obs
        for model in models:
            model.status ='PROCESSED'
            model.simulated_obs = pd.Series(1.0,index=rSubmit.config.obsNames())
        result = rSubmit.stdFunction(params2, raiseError=False)
        # 2 results so should be a 2 element vector
        self.assertEqual(result.shape[0], 2, 'Expected two element vector')
        # everything should be 1
        self.assertTrue(np.all(result == 1), 'Expected all to be missing')
        self.assertEqual(result.shape, (2, nobs), 'size not as expected')

        # no of models to run should be 4.. (ensemble member * param2)

        # now fake some obs!
        for m in models:
            m.simulated_obs = fake_fn(self.config,m.parameters)
            m.status = 'PROCESSED'
        # check if we run with df=True we get a dataframe...
        p = rSubmit.params().iloc[0, :].values  # first model
        result = rSubmit.stdFunction(p, df=True)  # should get back a 1 row dataframe.
        self.assertTrue(isinstance(result, pd.DataFrame), 'Should be dataframe')
        self.assertEqual(result.shape, (1, nobs), 'Size not expected')

        # check if we apply a transform that works. Only verify size and column names not values..
        trans = rSubmit.config.transMatrix()
        # will truncate this...
        trans = trans.iloc[0:5, :]
        # using params from prev test
        result = rSubmit.stdFunction(p, df=True, transform=trans)
        # check that result is 1x5 df with columns being 0..4
        expect_cols = np.arange(0, 5)
        nptest.assert_equal(expect_cols, result.columns)
        self.assertEqual(result.shape, (1, 5))

        # test if residual works.
        expect = rSubmit.stdFunction(p, df=True, scale=True) - rSubmit.config.targets(scale=True)
        result = rSubmit.stdFunction(p, df=True, scale=True, residual=True)
        self.assertTrue(expect.equals(result), 'Residual test failed')

        # test if sum of squares work
        # just use the previous test results...
        expect = (expect ** 2).sum(axis=1)
        result = rSubmit.stdFunction(p, df=True, scale=True, residual=True, sumSquare=True)
        self.assertTrue(expect.equals(result), 'SumSquare test failed')
        # and we are a series of size 1.
        self.assertEqual(result.size, 1, 'Size not as expected')

        # check that optclim_exceptions.runModelError is raised when we ask for new models above the limit.
        rSubmit.run_info['max_model_simulations']=len(rSubmit.model_index)
        with self.assertRaises(optclim_exceptions.runModelError):
            result = rSubmit.stdFunction(params*3)

    # test case for stdFunction

    def test_genOptFunction(self):
        """
        Test stdFunction which should return a function ready to drop into optimise/etc
         Will test by just running the fn and seeing we get a result!

        """

        # lets run  it
        configData = self.config
        configData.ensembleSize(2)
        rSubmit = runSubmit.runSubmit(copy.deepcopy(configData), 'optFn',
                                      rootDir=self.rootDir, refDir=self.refDir)
        params = rSubmit.config.beginParam()
        params = params.values
        nobs = len(rSubmit.config.obsNames())
        # test 1 -- result should be a vector of len nobs
        fn = rSubmit.genOptFunction(scale=True)
        with self.assertRaises(optclim_exceptions.runModelError):
            result = fn(params)  # should raise an runModelError exception

        # now run it.
        fk_fn = fake_run(rSubmit,scale=True)
        result = list(rSubmit.model_index.values())[0].simulated_obs
        self.assertEqual(result.shape, (nobs,), f'Expected {nobs} element array')
        pDict = dict(zip(rSubmit.config.paramNames(),params))
        expect = fk_fn(pDict).rename(result.name)
        pdtest.assert_series_equal(expect, result)

    @unittest.mock.patch.object(engine.sge_engine,'job_status', autospec=True, return_value='notFound')
    def test_runOptimized(self,mck):
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

        def run_all(config, name='run_test'):
            rSubmit = runSubmit.runSubmit(copy.deepcopy(config), name,
                                          rootDir=self.rootDir, refDir=self.refDir)
            try:
                rSubmit.runOptimized()
            except optclim_exceptions.runModelError:
                fake_funcn = fake_run(rSubmit)
            if np.unique(rSubmit.status()) != ['PROCESSED']:
                raise ValueError("Something odd")
            finalConfig = rSubmit.runOptimized()  # run it to get the actual final congig
            rSubmit.delete()  # clean up!
            return finalConfig,fake_funcn # and return the final config.

        configData = self.config

        # case 1 -- want to not set nensemble but do modify baseRunId and maxDigits

        configData.baseRunID('test1')
        configData.maxDigits(0)
        params = pd.Series(dict(CT=1e-4, EACF=0.5, ENTCOEF=3, ICE_SIZE=3e-5, RHCRIT=0.7, VF1=0.5, CW=2e-4))
        configData.optimumParams(optimum=params)
        finalConfig, fake_function = run_all(configData, name='run_opt')
        # and check what is to be created is as expected.
        std = finalConfig.standardParam()
        pDict = configData.optimumParams().to_dict()
        best = finalConfig.best_obs()
        expected = fake_function(pDict).rename(best.name)
        pdtest.assert_series_equal(best, expected)
        self.assertEqual(finalConfig.simObs().shape[0], 1)

        # WORKING TO HERE,
        # increase ensemble to four and shorten basename.
        configData.baseRunID('test')
        configData.maxDigits(1)
        configData.ensembleSize(4)
        finalConfig,fake_function = run_all(configData,  name='run_opt2')
        # and check what is to be created is as expected.
        pDict = configData.optimumParams().to_dict()
        best = finalConfig.best_obs()
        expected = fake_function(pDict).rename(best.name)
        pdtest.assert_series_equal(best, expected)
        self.assertEqual(finalConfig.simObs().shape[0], 4)

        optConfig = copy.deepcopy(configData)
        # set to min range.
        opt = optConfig.paramRanges().loc['minParam', :]
        optConfig.optimumParams(optimum=opt)
        finalConfig,fake_function = run_all(optConfig, name='run_opt3')
        # and check what is to be created is as expected.
        # and check what is to be created is as expected.
        pDict = optConfig.optimumParams().to_dict()
        best = finalConfig.best_obs()
        expected = fake_function(pDict).rename(best.name)
        pdtest.assert_series_equal(best, expected)
        self.assertEqual(finalConfig.simObs().shape[0], 4)

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

        # Set begin to max value.
        maxP = configData.paramRanges().loc['maxParam', :]
        begin = configData.beginParam(maxP)  # can truncate here using .iloc[0:x] if wanted.
        obs = configData.obsNames()
        nobs=len(obs)
        # for expected params should get back array close to 0
        # run DFOLS "naked"
        # set up DFOLS
        dfols_config = configData.DFOLS_config()
        dfols_config['maxfun'] = 75 # making this 100 seems to cause an extra evaluation and problems with the Jacobian.
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
        userParams.update(overwrite)
        userParams = configData.DFOLS_userParams(userParams=userParams)
        # and update the config...
        configData.DFOLS_config(dfols_config)
        configData.DFOLS_userParams(updateParams=userParams)

        ## directly run dfols
        np.random.seed(123456)  # make sure RNG is initialised to same value for first eval of dfols.
        # This should be the same seed as used in runDFOLS
        tgt = configData.targets(scale=scale)
        Tmat = configData.transMatrix(scale=scale)
        varParamNames = configData.paramNames()
        def fn_opt(param_v):  # function for optimisation,
            pDict = dict(zip(varParamNames, param_v))
            pDict.update(configData.fixedParams())
            sim_obs = fake_fn(configData,pDict)
            if scale:
                sim_obs *= configData.scales()
            sim_obs -= tgt
            sim_obs = sim_obs @ Tmat.T  # apply transform.
            return sim_obs

        prange = configData.paramRanges(paramNames=varParamNames)
        minP = prange.loc['minParam', :].values
        maxP = prange.loc['maxParam', :].values
        bounds = (minP, maxP)
        rangeP = maxP - minP
        expectparam = np.repeat(0.5, len(rangeP)) * rangeP + minP  # these param choices should give 0
        nptest.assert_allclose(fn_opt(expectparam), 0, atol=1e-4)
        solution = dfols.solve(fn_opt, begin.values,
                               objfun_has_noise=True,
                               bounds=bounds, scaling_within_bounds=True
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
        soln = pd.Series(solution.x,index=varParamNames).rename(f'Naked DFOLS')
        expectparam = pd.Series(expectparam,index=varParamNames).rename('best')
        df = pd.DataFrame([expectparam, soln])
        #    expect to be within 0.01% of the expected soln.
        nptest.assert_allclose(soln.values, expectparam.values, rtol=1e-4)

        # Run runDFOLS. WIll use fake_function to optimise directly.
        iterCount = 0
        rSubmit = runSubmit.runSubmit(configData, 'test_DFOLS',
                            rootDir=self.rootDir, refDir=self.refDir)
        # setup Submit object.
        while True:
            try:
                finalConfig = rSubmit.runDFOLS(scale=scale)  # run DFOLS
                break  # if we get to here then we are done.
            except optclim_exceptions.runModelError:  # Need to run some models which are "faked"
                fake_function = fake_run(rSubmit,scale=scale)
                iterCount += 1
                # expect nobs+1 on first iteration (parallel running)
                if iterCount == 1:
                    self.assertEqual(len(varParamNames) + 1, len(rSubmit.model_index),
                                     f'Expected to have {nobs + 1} models ran on iteration#1')




        # now compare results from DFOLS with those from "naked" dfols.
        best = finalConfig.optimumParams().rename('best')
        df = df.append(best.rename('DFOLS'))
        transJac = pd.DataFrame(finalConfig.dfols_solution().jacobian, columns=varParamNames, index=Tmat.index)

        result_transJac = finalConfig.transJacobian()
        info = finalConfig.get_dataFrameInfo(['diagnostic_info'])
        pdtest.assert_frame_equal(transJac, result_transJac, atol=1e-10)  # check Jacobian as stored is right
        #    expect to be within 0.01% of the expected soln. If random covariance done then this will be a
        # lot bigger
        pdtest.assert_series_equal(best, expectparam, rtol=1e-4)

        ## test that max_model_simulations overwited maxfun and generates a warning message.
        rSubmit.config.max_model_simulations(10) # set to 10
        rSubmit.config.DFOLS_config()['maxfun'] = 100 # set to something.
        with self.assertLogs('OPTCLIM.runSubmit', level="WARNING") as cm:
            finalConfig = rSubmit.runDFOLS(scale=scale)
        self.assertIsNotNone(finalConfig)
        self.assertEqual(len(cm),2,msg="Expected len 1 logs") # get two records. Not sure why! Dam logging
        self.assertIn("Overwriting value of maxfun=",cm[1][0])
        # expected no of evaluations should be max_model_simulations
        self.assertEqual(len(rSubmit.trace),rSubmit.config.max_model_simulations())
    @unittest.mock.patch.object(engine.sge_engine,'job_status', autospec=True, return_value='notFound')
    def test_runJacobian(self,mck):
        """
        test runJacobian!

        Drive with extreme cases for all  parameters should get dp towards centre (so easy to compute)
        Check jacobian from bare run and compare with runJacobian. Expect two goes at runJacobian and one submit.

        Test that optConfig works.
        """

        # Make the begin  values be max values (so we know perturbations take us to the center)

        # Modify the steps to be small but different.
        scale=True
        configData = self.config
        configData.beginParam(begin=configData.paramRanges().loc['maxParam',:])
        steps = configData.steps()
        steps.iloc[:] = np.arange(len(steps)) * 0.001 + 0.1  # want small but not equal steps.
        steps.loc['scale_steps'] = True
        configData.steps(steps=steps)  # set it in the configuration
        steps = configData.steps()  # get the steps we are actually using.
        configData.constraint(False)  # turn off constraint.
        prange = configData.paramRanges()  # param ranges
        optParams = prange.loc['maxParam']
        paramNames = optParams.index
        configData.paramNames(paramNames=paramNames)  # set the param names to reduced set.
        obsNames = configData.obsNames(add_constraint=False)  # get the obs names making sure constraint not used.

        def raw_jac(base,steps,scale:bool=False):
            if scale:
                scales = configData.scales()
            else:
                scales = 1.0
            ref = fake_fn(configData, base.to_dict())*scales
            jac_bare = []

            for p in steps.index:
                dd = base.copy()
                dd.loc[p] -=  steps.loc[p]
                delta = fake_fn(configData,dd.to_dict())*scales - ref
                delta = delta.rename(p)  # name it by param
                jac_bare.append(delta)
            jac_bare = pd.DataFrame(jac_bare)  # convert to dataframe
            # and divide by steps to get jacobian
            jac_bare = jac_bare.div(-steps, axis=0)
            # and apply linear transform
            Tmat = configData.transMatrix(scale=scale)
            jac_bare = jac_bare @ Tmat.T
            return jac_bare
        # all wrapped so we get a Series
        nparam = len(optParams)
        # compute what we expect --
        refParam = configData.beginParam()
        expect_jac = raw_jac(refParam,steps,scale=scale)
        rSubmit = runSubmit.runSubmit(configData, 'test_jac',
                                      rootDir=self.rootDir, refDir=self.refDir)
        # setup Submit object.
        while True:
            try:
                finalConfig = rSubmit.runJacobian(scale=scale)  # run Jacobian
                break
            except optclim_exceptions.runModelError:  # Need to run some models.
                fake_func = fake_run(rSubmit,scale=scale)
                # expect nparam+1 models all processed
                models = [model for model in rSubmit.model_index.values() if model.status == "PROCESSED"]
                nModels = len(models)
                # expect nparam+1 models
                self.assertEqual(nparam + 1, nModels, f'Expected to have {nparam + 1} models ran')

        # now check jac is what we expect...
        jac_run = finalConfig.transJacobian()
        nptest.assert_allclose(jac_run, expect_jac, atol=1e-9)  # round trip through json removes some precision.

        # check that passing in optConfig works and that have nparam +1 cases.
        optConfig = copy.deepcopy(self.config)
        # set optimum values to max,
        param_range = configData.paramRanges()
        opt = param_range.loc['minParam',:]+0.9*param_range.loc['rangeParam',:]
        optConfig.optimumParams(optimum=opt)
        expect_jac = raw_jac(opt, steps, scale=scale)
        rSubmit.delete() # clean up rSubmit -- no automatic deletion as want to keep disk stuff persistant.
        rSubmit = runSubmit.runSubmit(optConfig, 'test_opt_jac',
                                      rootDir=self.rootDir, refDir=self.refDir)
        while True:
            try:
                finalConfig = rSubmit.runJacobian(scale=scale)  # run Jacobian
                break
            except optclim_exceptions.runModelError:  # Need to run some models.
                fake_function = fake_run(rSubmit, scale=scale)
                models = [model for model in rSubmit.model_index.values() if model.status == "PROCESSED"]
                nModels = len(models)
                # expect nparam+1 models
                self.assertEqual(nparam + 1, nModels, f'Expected to have {nparam + 1} models ran')
        # now check jac is what we expect...
        jac_run = finalConfig.transJacobian()
        nptest.assert_allclose(jac_run, expect_jac, atol=1e-9)

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
        tgt = configData.targets(scale=scale)
        Tmat = configData.transMatrix(scale=scale)
        varParamNames = configData.paramNames()

        def fn_opt(param_v):  # function for optimisation,
            if param_v.ndim == 1:
                param_v = param_v.reshape(1, -1)
            nsim = param_v.shape[0]
            obs = []

            for indx in range(0,nsim):

                pDict = dict(zip(varParamNames, param_v[indx,:]))
                sim_obs = fake_fn(configData, pDict)
                if scale:
                    sim_obs *= configData.scales()
                sim_obs -= tgt
                sim_obs = sim_obs @ Tmat.T  # apply transform.
                obs.append(sim_obs)
            sim_obs = np.array(obs)
            return sim_obs

        prange = configData.paramRanges(paramNames=varParamNames)
        minP = prange.loc['minParam', :].values
        maxP = prange.loc['maxParam', :].values
        bounds = (minP, maxP)
        rangeP = maxP - minP
        expectparam = np.repeat(0.5, len(rangeP)) * rangeP + minP  # these param choices should give 0
        nptest.assert_allclose(fn_opt(expectparam), 0, atol=1e-4)
        # Set beginning values to max
        begin = configData.beginParam(configData.paramRanges().loc['maxParam', :])  # and also resets param names
        # run bare optimise.

        intCov = configData.Covariances(scale=scale)['CovTotal']*0.01
        # Scaling done for compatibility with optFunction.
        # need to transform intCov. errCov should be I after transform.
        intCov = Tmat.dot(intCov).dot(Tmat.T)
        # This is correct-- it is the internal covariance transformed
        optimise = dict(sigma=False,deterministicPerturb=20,maxIterations=20,alphas=[0.3,0.7,1])

        paramNames = configData.paramNames()
        nparam = len(paramNames)
        nObs = Tmat.shape[0]  # could be less than the "raw" obs depending on Tmat.
        # for this fake_fn (whcih cares nothing about the obs) we should set the step to 5% of the range.

        configData.steps(steps=configData.paramRanges().loc['rangeParam', :]*0.05)
        start = configData.beginParam(paramNames=paramNames)
        best, status, info = Optimise.gaussNewton(fn_opt, start.values,
                                                  configData.paramRanges(paramNames=paramNames).values.T,
                                                  configData.steps(paramNames=paramNames).values,
                                                  np.zeros(nObs), optimise,
                                                  cov=np.identity(nObs), cov_iv=intCov, trace=False)

        # expect to have converged and that best is close to expect
        self.assertEqual(status, 'Converged', msg='Expected to have converged')
        nptest.assert_allclose(best, expectparam, rtol=1e-4)  # close to 0.1%. Can probably do better by modifying covariance.
        # get out the jacobian for later comparision.
        jac = info['jacobian'][-1, :, :]  # bare right now...
        # now to run runGaussNewton.

        iterCount = 0
        nalpha = len(optimise['alphas'])
        rSubmit = runSubmit.runSubmit(configData, 'test_GN',
                                      rootDir=self.rootDir, refDir=self.refDir)
        # setup Submit object.
        while True:
            try:
                finalConfig = rSubmit.runGaussNewton(scale=scale,verbose=False)  # run Guass-Newton line-search.
                break  # if we get to here then we are done.
            except optclim_exceptions.runModelError:  # Need to run some models which are "faked"
                create_models = [model for model in rSubmit.model_index.values() if model.status == "CREATED"]
                nModels= len(create_models)
                fake_function = fake_run(rSubmit, scale=scale)
                iterCount += 1
                if iterCount == 1:
                    self.assertEqual(nparam + 1, len(rSubmit.model_index), f'Expected to have {nparam + 1} models ran on iteration#1')
                elif (iterCount % 2) == 0:
                    self.assertEqual(nalpha, nModels, f'Expected to have {nalpha} models ran on iteration# {iterCount}')
                else:
                    self.assertEqual(nparam, nModels, f'Expected to have {nparam} models ran on iteration# {iterCount}')

        # expect optimum value to be close to expected value.
        best = finalConfig.optimumParams()
        expect = pd.Series(expectparam, index=paramNames).rename('expect')
        print(pd.DataFrame([best, expect]))
        nptest.assert_allclose(best, expect, rtol=5e-4)
        nptest.assert_allclose(finalConfig.transJacobian(), jac,
                               atol=1e-10)  # check Jacobian as stored is right

        # check that setting maxIterations to 1 only has 1 iteration.
        shutil.rmtree(self.rootDir)
        configData.optimise(maxIterations=1)  # limit to 1 iteration
        rSubmit = runSubmit.runSubmit(configData, 'test_GN2',
                                      rootDir=self.rootDir, refDir=self.refDir)
        iterCount =0
        while True:
            try:
                finalConfig = rSubmit.runGaussNewton(scale=scale)  # run DFOLS
                break  # if we get to here then we are done.
            except optclim_exceptions.runModelError:  # Need to run some models which are "faked"
                create_models = [model for model in rSubmit.model_index.values() if model.status == "CREATED"]
                nModels = len(create_models)
                fake_function = fake_run(rSubmit, scale=scale)
                iterCount += 1

        self.assertEqual(finalConfig.GNparams().Iteration.size, 1, msg=f"Expected 1 iteration got {iterCount}")

    def test_dump_load(self):
        # test that dumping and loading work by dumping then loading and comparing the two objects.
        fp=self.rSubmit.config_path
        self.rSubmit.dump_config()
        nSubmit = self.rSubmit.load(fp)
        self.assertEqual(self.rSubmit,nSubmit)



if __name__ == "__main__":
    unittest.main()  # actually run the test cases
