#!/usr/bin/env python3
"""
Run generic algorithm (see OptClimVn2/runSubmit for what is available) with specified model and on
 cluster/supercomputer.
Been tested solidly for HadCM3 on Edinburgh's cluster Eddie.
Approach  relies on algorithm being deterministic. So repeatedly runs generating model simulations
as necessary. Only makes sense as function evaluation (running the model) very expensive. So
rereading them is really cheep!

Should be relatively easy to change to other model, optimisation algorithm and  other cluster/super-computers
general approach is that there is a class for specific model inheriting from a base class.
Main work needed is to define parameters and how they map to namelists (or what ever is needed to change model)
Algorithm runs and gets None if the model has not run. This error then gets trapped and
required models get generated.
xxxxSubmit handles the actual submission of the jobs, post processing to the cluster/super-computer
and submits itself.

Command line args:
jsonFile: path to jsonFile defining configuration.

do runAlgorithm -h to see what the remaining  command line arguments are.

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

from OptClimVn2 import StudyConfig, optClimLib, exceptions, runSubmit, config

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
                    help="Restart the optimisation by deleting all files and sub-directories of dir except json files (.json) in dir itself")
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

parser.add_argument("-o", "--optimise",
                    help='Name of JSON file providing configuration with optimisation. Used by runOptimised & '
                         'runJacobian')

helpStr = """Behaviour for model with no observations. 
                         Choices are fail (default), continue (continue run and submit), 
                         perturb (pertrub run, restart and submit), 
                         pertrubc (pertrub run, continue and submit),
                         or clean (remove directory and continue as normal).
                         These are all passed into Submit but no "new" simulations will be submitted. 
                         Just the failed simulations."""
parser.add_argument("--noobs", help=helpStr,
                    default='fail',
                    choices=['fail', 'continue', 'perturb', 'perturbc','clean'])
args = parser.parse_args()
verbose = args.verbose
resubmit = args.noresubmit
dryRun = args.dryrun
testRun = args.test
jsonFile = pathlib.Path(os.path.expanduser(os.path.expandvars(args.jsonFile)))
monitor = args.monitor
genNew = not args.nonew
restart = args.restart  # do we want to restart?

configData = StudyConfig.readConfig(filename=jsonFile, ordered=True)  # parse the jsonFile.
rootDir = None

restartCMD = [os.path.realpath(sys.argv[0]), args.jsonFile]  # generate restart cmd

if args.dir is not None:
    rootDir = os.path.expanduser(os.path.expandvars(args.dir))  # directory defined so overwrite rootDir
    rootDir = pathlib.Path(rootDir)  # convert to a Path object.
    restartCMD.extend(['--dir', args.dir])
else:  # code from Submit -- this whole block should go in there along with restart handling
    rootDir = pathlib.Path.cwd() / configData.name()  # default path

if args.monitor:
    restartCMD.extend(['--monitor'])

optConfig = None  # default is no extra optimisation config file
if args.optimise:
    restartCMD.extend([f'--optimise {args.optimise}'])
    optConfig = StudyConfig.readConfig(args.optimise)
if verbose:
    print("Running from config %s named %s" % (jsonFile, configData.name()))
    restartCMD.extend(['--verbose'])
    # TODO modify code to use logging and provide a logging file
# having logging on.
logging.basicConfig(level=logging.DEBUG, format='%(message)s')  # detailed info: show every function evaluation

if not resubmit: restartCMD = None  # nothing to resubmit.

fakeFn = None
if testRun:
    fakeFn = config.fake_fn

# work out what final json file is called regardless of why run finished.
rootDiagFiles, ext = os.path.splitext(os.path.basename(jsonFile))
finalJsonFile = rootDir / (rootDiagFiles + "_final" + ext)
monitorFile = rootDir / (rootDiagFiles + "_monitor.png")

##############################################################
# Main block of code
#  Now we actually run the Algorithm
##############################################################

# common stuff across all algorithms


doRun = True  # keep runnign until done. If no fake fn will breakout from this loop to finish.
iterCount = 0  # for testing print out iterCount

while doRun:
    np.random.seed(123456)  # init RNG though probably should go to the runXXX methods.
    # setup MODELRUN
    MODELRUN = runSubmit.runSubmit(configData,
                                   configData.modelFunction(config.modelFunctions),  # model function
                                   configData.submitFunction(config.submitFunctions),  # submit function
                                   fakeFn=fakeFn, rootDir=rootDir, verbose=verbose,
                                   readOnly=args.readOnly, noObs=args.noobs, restart=restart)
    restart = False  # subsequently do not want to restart (i.e. clean up dir)

    try:
        minModels = MODELRUN.rerunModels()
        if len(minModels) > 0:  # got some minimal models to run so trigger error
            raise exceptions.runModelError(
                '(re)ran cases with no observations')  # bit of  a hack. Will take us to the end

        algorithmName = configData.optimise()['algorithm'].upper()
        if algorithmName == 'DFOLS':
            finalConfig = MODELRUN.runDFOLS(scale=True)
        elif algorithmName == 'PYSOT':
            # pySOT -- probably won't work without some work.
            finalConfig = MODELRUN.runPYSOT(scale=True)
        elif algorithmName == 'GAUSSNEWTON':
            finalConfig = MODELRUN.runGaussNewton(scale=True)
        elif algorithmName == 'JACOBIAN':
            # compute the Jacobian.
            finalConfig = MODELRUN.runJacobian(optConfig=optConfig)
        elif algorithmName == 'RUNOPTIMISED':  # run optimised case through configuration in JSON file.
            finalConfig = MODELRUN.runOptimized(optConfig=optConfig)
        else:
            raise Exception(f"Don't know what to do with Algorithm: {algorithmName}")

        doRun = False  # we have finished so can exit. Though only makes sense when have a fake fn.
    except exceptions.runModelError:  # error which triggers need to run more models.
        # run more models
        status, nModels, finalConfig = MODELRUN.submit(resubmit=restartCMD, dryRun=dryRun)
        if not status:
            raise Exception(f"Some problem. status = {status}")
    # end of try/except. Now clean up.
    finalConfig.save(filename=finalJsonFile)  # save the (updated) configuration file.
    if fakeFn is None:  # no fake fn so time to exit. This is "normal" behaviour.
        break  # exit the run for ever loop as no more runs should be submitted on this go.
    else:  # we have a fake function so keep going-- this is test mode
        iterCount += 1
        print(f"On iteration {iterCount} submitted {nModels} models")

# optionally produce monitoring picture. Only doing at the end even when fakeFn active.
if args.monitor:
    finalConfig.plot(monitorFile=monitorFile)
