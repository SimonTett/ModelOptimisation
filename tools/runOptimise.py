#!/usr/bin/env python3
"""
THIS WILL NO LONGER WORK AND IS BEING LEFT FOR A WHILE!
Run DFO-LS or gauss-Newton optimisation using HadCM3 on Eddie.
This relies on algorithm being deterministic. So repeatedly runs generating model simulations
as necessary. Only makes sense as function evaluation (running the model) very expensive. So
rereading them is really cheap!

Should be relatively easy to change to other model, optimisation algorithm and  other cluster/super-computers
general approach is that there is a class for specific model inheriting from a base class.
Main work needed is to define parameters and how they map to namelists (or what ever is needed to change model)
Algorithm runs and gets None if the model has not run. This error then gets trapped and
required models get generated.
xxxxSubmit handles the actual submission of the jobs, post processing to the cluster/super-computer
and submits itself.

Command line args:
jsonFile: path to jsonFile defining configuration.

example run test case:
import tempfile
tdir= tempfile.TemporaryDirectory()

%run -i tools/runOptimise --restart  --test -d $tdir.name Configurations/example.json
# then repeat until done.
%run -i tools/runOptimise --test -d $tdir.name Configurations/example.json
# FIXME when continuing if did not have parameters set first time they are ignored.
# I think this is reasonable -- if they are not set then whatever there gets used.
# but if they are set then this overwrites. Probably better to write a adhoc HadCM3 case which
# reads from the namelists and then sets the configuration.
TODO: Modify to allow "random" running to identify which new cases can be run in parallel. Think this involves some magic with optimisation. Perhaps best to wrap up in an object using similar approach to scipy.optimize.

TODO: Add test cases. Not sure how to do this. Only obvious way is to compare with earlier runs. See config files in  testTools
"""
import os

# often this script runs on a headless display.
# so need to set up maplotlib (if we plot) for that case.
# this bit of code needs to run before any other matplotlib code.
import matplotlib

display = os.environ.get("DISPLAY")
if os.name == 'nt':
    display = 'windows'  # fake a display
if (display is None) or (not display):  # no display defined
    matplotlib.use('Agg')  # needed for "headless" nodes

import argparse  # parse command line arguments
import logging
import os
import shutil
import stat
import sys
import pathlib
import numpy as np

np.seterr(all='raise')  # define before your code.
import pandas as pd


import config
import dfols  # optimisation ...
from OptClimVn2 import Optimise, StudyConfig, Submit, optClimLib


finished = None # set true once algorithm has finished


# try and import HadCM3Coupled which provides specialist functions for
# transient climate response and equilibrium climate sensitivity.
# These run two model simulations. If you don't need this then
# don't worry if the import fails.
try:
    import HadCM3Coupled  # sets config.XXXX lookup tables with HadCM3Coupled functions.
except ImportError:
    print("Failed to load HadCM3Coupled")



## main script

## set up command line args

parser = argparse.ArgumentParser(description="Run study")
parser.add_argument("-d", "--dir", help="path to root directory where model runs will be created")
parser.add_argument("jsonFile", help="json file that defines the study")
parser.add_argument("-r", "--restart", action='store_true',
                    help="Restart the optimisation by deleting all files except json file")
parser.add_argument("-v", "--verbose", action='store_true', help="Provide Verbose information")
parser.add_argument("-n", "--noresubmit", action='store_false',
                    help="If set then do not resubmit this script. Good for testing")
parser.add_argument("--dryrun", action='store_true',
                    help="if set do not submit any jobs but do create directories. Good for testing")
parser.add_argument("--readOnly", action='store_true', help="read data but do not create directories or submit jobs")
parser.add_argument("--nonew", action='store_true', help="If set fail if generate a new model.")
# TODO merge dryrun and noresubmit together.
parser.add_argument("-t", "--test", action='store_true',
                    help='If set run fake codes rather than submitting models.')
parser.add_argument("-m", "--monitor", action='store_true', help='Producing monitoring plot after running')
parser.add_argument("-V", "--verify", action='store_true',
                    help='Verify test results by comparision with old case -- hardwired for test cases and only available for some algorithms ')

helpStr = """Behaviour for model with no observations. 
                         Choices are fail (default), continue (continue run and submit), 
                         perturb (pertrub run, restart and submit), or clean (remove directory and continue as normal).
                         These are all passed into Submit but no "new" simulations will be submitted. 
                         Just the failed simulations."""
parser.add_argument("--noobs",  help=helpStr,
                    default='fail',
                    choices=['fail', 'continue', 'perturb', 'clean'])
args = parser.parse_args()
verbose = args.verbose
resubmit = args.noresubmit
dryRun = args.dryrun
testRun = args.test
jsonFile = os.path.expanduser(os.path.expandvars(args.jsonFile))
monitor = args.monitor
genNew = not args.nonew

configData = StudyConfig.readConfig(filename=jsonFile, ordered=True)  # parse the jsonFile.
rootDir = None

restartCMD = [os.path.realpath(sys.argv[0]), args.jsonFile]  # generate restart cmd

if args.dir is not None:
    rootDir = os.path.expanduser(os.path.expandvars(args.dir))  # directory defined so overwrite rootDir
    rootDir = pathlib.Path(rootDir) # convert to a Path object.
    restartCMD.extend(['--dir', args.dir])
else: # code from Subvmit -- this whole block should go in there along with restart handling
    rootDir = pathlib.Path.cwd()/configData.name() # default path

if args.monitor:
    restartCMD.extend(['--monitor'])
if verbose:
    print("Running from config %s named %s" % (jsonFile, configData.name()))
    restartCMD.extend(['--verbose'])
    # TODO modify code to use logging and provide a logging file
 # having logging on.
logging.basicConfig(level=logging.DEBUG, format='%(message)s')  # detailed info: show every function evaluation

if not resubmit: restartCMD = None  # nothing to resubmit.
if args.restart and (rootDir is not None) and rootDir.is_dir():  # starting anew.
    # TODO move this to Submit.
    if os.path.split(rootDir)[-1] == 'Configurations':
        print("You will delete configurations -- aborting")
        raise Exception("Attempted to delete configurations")
    # go and clean all directories  by removing everything EXCEPT args.jsonFile
    # algorithm -- iterate over all files in rootDir
    #  if file is a file and not jsonFile delete it,   if file is a dir delete it
    for p in os.listdir(rootDir):
        fp = os.path.join(rootDir, p)
        if os.path.isfile(fp) and os.path.basename(p) != os.path.basename(jsonFile):
            if verbose: print("Deleting %s" % fp)
            os.chmod(fp, stat.S_IWRITE)
            os.remove(fp)  # remove the file
        elif os.path.isdir(fp):
            if verbose: print("Deleting %s and contents" % fp)
            shutil.rmtree(fp, onerror=optClimLib.errorRemoveReadonly)

fakeFn = None
if testRun:
    fakeFn = configData.fakeFunction(config.fakeFunctions)  # function for fake submit the models.

# setup MODELRUN

# rewrite rules to make eddie data work on Simon T's laptop!
renameRefDir = {'/exports/csce/eddie/geos/groups/OPTCLIM/software/optimise2018/Configurations/xnmea': 'ctl',
                '/exports/csce/eddie/geos/groups/OPTCLIM/software/optimise2018/Configurations/xnmed': '4xCO2',
                r'm:\analyse\optclim\currentCode\Configurations\xnmea': 'ctl',
                r'm:\analyse\optclim\currentCode\Configurations\xnmed': '4xCO2'
                }


MODELRUN = Submit.ModelSubmit(configData,
                              configData.modelFunction(config.modelFunctions),  # model function
                              configData.submitFunction(config.submitFunctions),  # submit function
                              fakeFn=fakeFn, rootDir=rootDir, verbose=verbose,
                              readOnly=args.readOnly, noObs=args.noobs)




#TODO -- fix  rename as doesn't currently work when submitting jobs.
nObs = len(MODELRUN.obsNames())
varParamNames = configData.paramNames()  # extract the parameter names if have them
optimise = configData.optimise().copy()  # get optimisation info


start = configData.beginParam(paramNames=varParamNames)
verifyDir = os.path.join(os.environ['OPTCLIMTOP'], 'tools', 'testTools')
# work out what final json file is called regardless of why run finished.
rootDiagFiles, ext = os.path.splitext(os.path.basename(jsonFile))
finalJsonFile = MODELRUN.rootDir/(rootDiagFiles + "_final" + ext)
monitorFile =MODELRUN.rootDir/(rootDiagFiles + "_monitor.png")
verifyFile = os.path.join(verifyDir, 'expect_' + rootDiagFiles + "_final" + ext)

##############################################################
# Main block of code
#  Now we actually run the optimisation code.
##############################################################

# common stuff across all algorithms

np.random.seed(123456)  # init RNG
algorithmName = optimise['algorithm'].upper()
tMat = configData.transMatrix()
try:
    minModels = MODELRUN.rerunModels()
    if len(minModels) > 0:  # got some minimal models to run so trigger error
        raise Submit.runModelError('(re)ran cases with no observations') # bit of  a hack. Will take us to the end
    if algorithmName == 'DFOLS':
        dfols_config=configData.DFOLS_config()
        # general configuration of DFOLS -- which can be overwritten by config file
        userParams = {'logging.save_diagnostic_info': True,
                      'logging.save_xk': True,
                      'noise.quit_on_noise_level': True,
                      'general.check_objfun_for_overflow': False,
                      'init.run_in_parallel': True,
                      'general.check_objfun_for_overflow': False,
                      'interpolation.throw_error_on_nans':True, # make an error happen!
                      }
        prange = (configData.paramRanges(paramNames=varParamNames).loc['minParam', :].values,
                  configData.paramRanges(paramNames=varParamNames).loc['maxParam', :].values)
        # update the user parameters from the configuration.
        userParams = configData.DFOLS_userParams(userParams=userParams)

        optFn = MODELRUN.genOptFunction(transform=tMat, residual=True,raiseError=False,scale=True) #TODO determine if scale needs to be set.
        #seem to have lost multiple evaluations here.. Think it is a DFOLS feature as have same problem with "old" code.
        # will ask Lindon about this. Actaully don'tthink so. Downgraded to same version as eddie and still have the
        # same problem...
        np.seterr(all='raise')
        try:
            solution = dfols.solve(optFn, start.values,
                               objfun_has_noise=True,
                               bounds=prange,scaling_within_bounds=True,
                               maxfun=dfols_config.get('maxfun', 100),
                               rhobeg=dfols_config.get('rhobeg', 1e-1),
                               rhoend=dfols_config.get('rhoend', 1e-3),
                               user_params=userParams)
        except np.linalg.linalg.LinAlgError:
            raise(Submit.runModelError("dfols failed with lin alg error"))

        # code here will be run when DFOLS has completed. It mostly is to put stuff in the final JSON file
        # so can easily be looked at for subsequent analysis.
        # some of it could be done even if DFOLS did not complete.
        if solution.flag not in (solution.EXIT_SUCCESS, solution.EXIT_MAXFUN_WARNING):
            print("dfols failed with flag %i error : %s" % (solution.flag, solution.msg))
            raise Exception("Problem with dfols")

        # need to wrap best soln.
        finalConfig = MODELRUN.runConfig(Config)  # get final runInfo
        best = pd.Series(solution.x, index=varParamNames).rename(finalConfig.name())
        best_obs = solution.resid
        best_obs = pd.Series(best_obs,index=range(0,len(best_obs))).rename(finalConfig.name())
        finalConfig.best_obs(best_obs = best_obs)
        jac = pd.DataFrame(solution.jacobian, columns=varParamNames) # not transforming data
        finalConfig.jacobian(jacobian=jac) # store the jacobian.
        finalConfig.optimumParams(**(best.to_dict()))  # write the optimum params
        # need to put in the best case -- which may not be the best evaluation as DFOLS ignores "burn in"
        solution.diagnostic_info.index = range(0, solution.diagnostic_info.shape[0])  # this does not include "burn in" evals
        info = finalConfig.diagnosticInfo(diagnostic=solution.diagnostic_info)

        print("DFOLS completed: Solution status: %s" % (solution.msg))
    elif algorithmName == 'PYSOT':
        # pySOT -- probably won't work without some work.
        optFn = MODELRUN.genOptFunction(transform=tMat, residual=True) # need scale??
        import pySOT
        from pySOT.experimental_design import SymmetricLatinHypercube
        from pySOT.strategy import SRBFStrategy, DYCORSStrategy  # , SOPStrategy
        from pySOT.surrogate import RBFInterpolant, CubicKernel, LinearTail, \
            SurrogateUnitBox  # will not work anymore as SurrogateUnitBox not defined.
        from poap.controller import SerialController
        from pySOT.optimization_problems import OptimizationProblem
        # Wrapper written for pySOT 0.2.2 (installed from conda-forge)
        # written by Lindon Roberts
        # Based on
        # https://github.com/dme65/pySOT/blob/master/pySOT/examples/example_simple.py
        # Expect optimise parameters:
        #  - maxfun: total number of evaluations allowed, default 100
        #  - initial_npts: number of initial evaluations, default 2*n+1 where n is the number of variables to optimise
        pysot_config = optimise.get('pysot', {})

        # Light wrapper of objfun for pySOT framework
        class WrappedObjFun(OptimizationProblem):
            def __init__(self):
                self.lb = configData.paramRanges(paramNames=varParamNames).loc['minParam', :].values  # lower bounds
                self.ub = configData.paramRanges(paramNames=varParamNames).loc['maxParam', :].values  # upper bounds
                self.dim = len(self.lb)  # dimensionality
                self.info = "Wrapper to DFOLS cost function"  # info
                self.int_var = np.array([])  # integer variables
                self.cont_var = np.arange(self.dim)  # continuous variables
                self.dfols_residual_function = optFn

            def eval(self, x):
                # Return same cost function as DFO-LS gets
                residuals = self.dfols_residual_function(x)  # i.e. if DFO-LS asked for the model cost at x, it would get the vector "residuals"
                dfols_cost = np.dot(residuals, residuals)  # sum of squares (no constant in front) - matches DFO-LS internal cost function
                return dfols_cost

        data = WrappedObjFun()  # instantiate wrapped objective function

        # Initial design of points
        slhd = SymmetricLatinHypercube(dim=data.dim, num_pts=pysot_config.get('initial_npts', 2*data.dim + 1))

        # Choice of surrogate model (cubic RBF interpolant with a linear tail)
        rbf = SurrogateUnitBox(RBFInterpolant(dim=data.dim, kernel=CubicKernel(), tail=LinearTail(data.dim)), lb=data.lb, ub=data.ub)

        # Use the serial controller (uses only one thread), SRBF strategy to find new points
        controller = SerialController(data.eval)
        strategy = pysot_config.get('strategy', 'SRBF')
        maxfun = pysot_config.get('maxfun', 100)
        if strategy == 'SRBF':
            controller.strategy = SRBFStrategy(max_evals=maxfun, opt_prob=data, exp_design=slhd, surrogate=rbf)
        elif strategy == 'DYCORS':
            controller.strategy = DYCORSStrategy(max_evals=maxfun, opt_prob=data, exp_design=slhd, surrogate=rbf)
        else:
            raise RuntimeError("Unknown pySOT strategy: %s (expect SRBF or DYCORS)" % strategy)

        # Run the optimization
        result = controller.run()



        # code here will be run when PYSOT has completed. It is mostly is to put stuff in the final JSON file
        # Gather key outputs: optimal x, optimal objective value, number of objective evaluations used
        xmin = result.params[0]
        fmin = result.value
        nf = len(controller.fevals)

        # need to wrap best soln xmin.
        finalConfig = MODELRUN.runConfig(Config)  # get final runInfo
        best = pd.Series(xmin, index=varParamNames)
        finalConfig.optimumParams(**(best.to_dict()))  # write the optimum params
        print("PYSOT completed")
    elif algorithmName == 'GAUSSNEWTON':
        # extract internal covariance and transform it.
        intCov = configData.Covariances(trace=verbose, scale=True)['CovIntVar']
        # Scaling done for compatibility with optFunction.
        # need to transform intCov. errCov should be I after transform.
        tMat = configData.transMatrix()
        intCov = tMat.dot(intCov).dot(tMat.T)
        optimise['sigma'] = False  # wrapped optimisation into cost function.
        optimise['deterministicPerturb'] = True  # deterministic perturbations.
        optFn = MODELRUN.genOptFunction(transform=tMat) # think only need tMat. What about scaling???
        best, status, info = Optimise.gaussNewton(MODELRUN.optFunction, start.values,
                                                  configData.paramRanges(paramNames=varParamNames).values.T,
                                                  configData.steps(paramNames=varParamNames).values,
                                                  np.zeros(nObs), optimise,
                                                  cov=np.identity(nObs), cov_iv=intCov, trace=verbose)

        finalConfig = MODELRUN.runConfig(Config)  # get final runInfo
        jacobian = finalConfig.GNjacobian(info['jacobian'])
        hessian = finalConfig.GNhessian(info['hessian'])
        params = finalConfig.GNparams(info['bestParams'])
        cost = finalConfig.GNcost(info['err_constraint'])
        alpha = finalConfig.GNalpha(info['alpha'])
        best = pd.Series(best, index=finalConfig.paramNames(),
                         name=finalConfig.name())  # wrap best result as pandas series

        print("All done with status: %s " % (status))
        # print best param values, cost and alpha values
        # first  add cost and alpha to params array
        p2 = params.to_pandas().assign(cost=cost, alpha=alpha)
        with pd.option_context('max_rows', None, 'max_columns', None, 'precision', 2,
                               'expand_frame_repr', True, 'display.width', 120):
            print(p2)
        best = finalConfig.optimumParams(**(best.to_dict()))  # write the optimum params

    elif algorithmName == 'JACOBIAN':
        # compute the Jacobian.
        optFn = MODELRUN.genOptFunction() # do we need scaling??
        jacobian = Optimise.runJacobian(MODELRUN.optFunction, start, configData.steps(paramNames=varParamNames),
                                        configData.paramRanges(paramNames=varParamNames),
                                        obsNames=configData.obsNames(), returnVar=True)
        # store  result
        finalConfig = MODELRUN.runConfig(Config)  # get final runInfo
        jacobian = finalConfig.runJacobian(jacobian)
        print("All done running Jacobian")
    elif algorithmName == 'RUNOPTIMISED': # run optimised case through configuration in JSON file.
        finalConfig = MODELRUN.runOptimized()
        #raise NotImplementedError("Not yet implemented runOptimised yet")


    else:
        raise Exception(f"Don't know what to do with Algorithm: {algorithmName}")


    # print out some useful information from algorithm
    # put in a try statement to pick up errors.
    finished = True # we have finished.
    try:
        start = finalConfig.beginParam()
        start.name = 'start'
        best = finalConfig.optimumParams()
        best.name = 'best'
        # lookup the label for the best config.
        # add ensembleMember 0 -- arbitrary and with ensembles no run will correspond to the best case..
        best.loc['ensembleMember']=0
        try:
            bestID = MODELRUN.model(params=best).name()
        except AttributeError:
            print("got an attribute error.Something wrong with model")
            breakpoint()
        finalConfig.bestEval = bestID
        # include cost in if we have it!
        try:  # could go wrong if cost defined badly!
            best['cost'] = cost.loc[bestID] # I think this is in error...Though do not think anything done with best['cost']
            start['cost'] = cost.iloc[0]
        except NameError:
            pass  # we don't have costs.
        with pd.option_context('max_rows', None, 'max_columns', 20, 'precision', 2):
            print("start to best params are: ")
            print(pd.DataFrame([start, best]))
    except KeyError:
        print("Something missing so not printing minimal information out")


    # optionally verify results
    if args.verify:
        verifyConfig = StudyConfig.readConfig(verifyFile, ordered=True)
        # runs should be the same!
        tol = 1e-12
        if (np.any(np.abs(verifyConfig.parameters() - finalConfig.parameters()) > tol)):
            raise Exception("Verification failed with parameter differences")
        # and so should costs
        if (np.any(np.abs(verifyConfig.cost() - finalConfig.cost()) > 2e-1)):
            raise Exception("Verification failed with cost differences")

        print("Verified for Algorithm %s both cost and parameters" % algorithmName)
except (Submit.runModelError):  # error which triggers need to run more models.
    # update configData info and save to temp file.

    # work out fn for submission.

    status, nModels, finalConfig = MODELRUN.submit(resubmit=restartCMD, dryRun=dryRun)
    if not status:
        raise Exception("Some problem...")

# all end of run stuff
finalConfig.save(filename=finalJsonFile)  # save the (updated) configuration file.
# optionally produce monitoring picture.
if args.monitor:
    #finalConfig.plot(monitorFile=monitorFile)
    pass
