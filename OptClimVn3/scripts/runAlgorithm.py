#!/usr/bin/env python3
"""
Run generic algorithm (see OptClimVn3/runSubmit for what is available) with specified model and on
 cluster/supercomputer.
Approach  relies on algorithm being deterministic. So repeatedly runs generating model simulations
as necessary. Only makes sense as function evaluation (running the model) very expensive. So
rereading them is relatively cheap!

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
import functools
import os
import sys
import pathlib
import numpy as np
import exceptions
import runSubmit
import StudyConfig

from Models import *  # all models.
import genericLib
import logging
from Models.HadCM3 import  HadCM3
from Models.simple_model import simple_model
from Models.Model import Model  # for generic stuff to do with models.
## main script

## set up command line args

parser = argparse.ArgumentParser(description="Run study")
parser.add_argument("-d", "--dir", help="path to root directory where model runs will be created")
parser.add_argument("jsonFile", help="json file that defines the study")
parser.add_argument("--delete", action='store_true',
                    help="Delete the configuration")
parser.add_argument("-v", "--verbose", action='count', default=0,
                    help="level of logging info level= 1 = info, level = 2 = debug ")
parser.add_argument("--dryrun", action='store_true',
                    help="if set do not submit any jobs but do instantiate models. Good for testing")
parser.add_argument("--readOnly", action='store_true', help="read data but do not instantiate or submit jobs.")
parser.add_argument("-t", "--test", action='store_true',
                    help='If set run fake function rather than submitting models.')
parser.add_argument("-m", "--monitor", action='store_true', help='Producing monitoring plot after running')

helpStr = """Behaviour for models that failed. Choices are:
                fail (default), 
                continue (continue run with no changes)), 
                perturb (perturb run, restart run), 
                perturbc (perturb run, continue run),
                delete (delete model).
            These cases will be submitted (unless delete or fail are choice)"""
parser.add_argument("--fail", help=helpStr,
                    default='fail',
                    choices=['fail', 'continue', 'perturb', 'perturbc', 'delete'])
args = parser.parse_args()
verbose = args.verbose
dry_run = args.dryrun
read_only = args.readOnly
testRun = args.test
jsonFile = pathlib.Path(os.path.expanduser(os.path.expandvars(args.jsonFile)))
delete = args.delete
monitor = args.monitor
fail = args.fail

if verbose == 1:
    logging.basicConfig(level=logging.INFO)
elif verbose > 1:
    logging.basicConfig(level=logging.DEBUG)
else:
    pass

configData = StudyConfig.readConfig(filename=jsonFile)  # parse the jsonFile.

args_not_for_restart = ['-r', '--restart']  # arguments to be removed from the restart cmd
restartCMD = [arg for arg in sys.argv if arg not in args_not_for_restart]  # generate restart cmd.
if dry_run or read_only:
    restartCMD = None  # no restarting!

logging.info("Running from config %s named %s" % (jsonFile, configData.name()))

if args.dir is not None:
    rootDir = Model.expand(args.dir)  # directory defined so set rootDir
else:  # set rootDir to cwd/name
    rootDir = pathlib.Path.cwd() / configData.name()  # default path

config_path = rootDir / (configData.name() + ".scfg")

if testRun:
    fakeFn = functools.partial(genericLib.fake_fn, configData)
    fakeFn.__name__ = 'partial genericLib.fake_fn with configData'
else:
    fakeFn = None

# work out final json file and possible monitor_file
final_JSON_file = rootDir / (jsonFile.stem + "_final.json")
monitor_file = rootDir / (jsonFile.stem + "_monitor.png")

##############################################################
# Main block of code
#  Now we actually run the Algorithm
##############################################################

# common stuff across all algorithms
iterCount = 0  # for testing print out iterCount
nModels = 0
wantCost = True
rSUBMIT = None  # set it to None
if config_path.exists():  # config file exists. Read it in.
    logging.info(f"Reading status from {config_path}")
    rSUBMIT = runSubmit.runSubmit.load_SubmitStudy(config_path)
    if not isinstance(rSUBMIT, runSubmit.runSubmit):
        raise ValueError(f"Something wrong")

    if delete:  # delete the config
        logging.info(f"Deleting existing config {rSUBMIT}")
        rSUBMIT.delete()  # should clean dir.
        rSUBMIT = None # remove it.

if rSUBMIT is None:# no configuration exists. So create it.
    rSUBMIT = runSubmit.runSubmit(configData,  rootDir=rootDir, config_path=config_path)
    logging.debug(f"Created new runSubmit {rSUBMIT}")



# We might  have runs to do so check that and run them if so.
if not (dry_run or read_only):  # not dry running or read only.
    # so first deal with failed models and then models that are already instantiated and so need running.
    # This happens if not all models that were instantiated were submitted.
    failed_models = rSUBMIT.failed_models()
    if len(failed_models):  # Some runs failed. Use fail to decide what to do
        logging.info(f"{failed_models} models failed. {rSUBMIT}")
        if fail == 'fail':
            raise ValueError(f"{rSUBMIT} has FAILED models. Use --fail choice ")
        for model in failed_models:
            logging.debug(f"Dealing with fail = {fail} for model {model}")
            if fail in ['perturb', 'perturbc']:
                model.perturb()  # perturb  model
            if fail in ['perturbc', 'continue']:  # model to continue
                model.status = 'CONTINUE'
            if fail == 'delete':  # delete model
                rSUBMIT.delete_model(model)
        if fail in ['perturbc', 'continue']:
            logging.info(f"Continuing {len(rSUBMIT.models_to_continue())}")
    #  submit models and exit -- only those that are submittable will be submitted.
    nModels = rSUBMIT.submit_all_models(restartCMD, fake_fn=fakeFn)
    # this handles both models that are instantiated or those that need continuing.
    # see submit_all_models for details.
    if nModels > 0:  # submitted some models
        logging.info(f"Submitted {nModels}. rSubmit: {rSUBMIT}")
        exit(0)  # just exit.

# check status is only PROCESSED.
status = rSUBMIT.status()
if np.any(status != 'PROCESSED'):
    raise ValueError(f"Have unexpected status rSUBMIT:{rSUBMIT}")


algorithmName = configData.optimise()['algorithm'].upper()
logging.debug(f"Algorithm is {algorithmName}")
if algorithmName in ['GUASSNEWTON', 'JACOBIAN']:
    wantCost = False
else:
    wantCost = True
finalConfig = None

while True:
    try:  # run an algorithm iteration.
        np.random.seed(123456)  # init RNG though probably should go to the runXXX methods.
        if algorithmName == 'DFOLS':
            finalConfig = rSUBMIT.runDFOLS(scale=True)
        elif algorithmName == 'PYSOT':
            # pySOT -- probably won't work without some work.
            finalConfig = rSUBMIT.runPYSOT(scale=True)
        elif algorithmName == 'GAUSSNEWTON':
            finalConfig = rSUBMIT.runGaussNewton(scale=True)
        elif algorithmName == 'JACOBIAN':
            # compute the Jacobian.
            finalConfig = rSUBMIT.runJacobian()
        elif algorithmName == 'RUNOPTIMISED':  # run optimised case through configuration in JSON file.
            finalConfig = rSUBMIT.runOptimized()
        else:
            raise ValueError(f"Don't know what to do with Algorithm: {algorithmName}")
        break  # we have finished running algorithm so can exit and go to final clear up.
    except exceptions.runModelError:  # error which triggers need to instantiate and run more models.
        if read_only:
            logging.info(f"read_only -- exiting")
            break  # exit the loop -- we are done as in read_only mode.
        rSUBMIT.instantiate()  # instantiate all cases that need instantiation.
        logging.info(f"Instantiated {rSUBMIT}")
        if dry_run:  # nothing gets submitted or faked. So exit
            logging.info(f"dry_run -- exiting")
            break
        nModels = rSUBMIT.submit_all_models(restartCMD, fake_fn=fakeFn) # this also saves the config.
        finalConfig = rSUBMIT.runConfig(scale=True, add_cost=wantCost) # generate final configuration
        iterCount += 1
        logging.info(f"On iteration {iterCount} submitted {nModels} models")
        try:
            logging.info(f"Last cost is {float(finalConfig.cost().iloc[-1])}")
        except IndexError: # no cost.
            pass
        if fakeFn is None:  # no fake fn so time to exit. This is "normal" behaviour.
            logging.info("Exiting")
            break  # exit the run forever loop as no more runs should be submitted on this go.
        else: # reload the configuration (and all models).
            # This necessary as writing out/reading in changes (slightly) the floating point value of some values
            # which in turn changes the way the algorithms behave.
            rSUBMIT = runSubmit.runSubmit.load_SubmitStudy(config_path)

    # end of try/except.

# Deal with final stuff
rSUBMIT.dump_config()  # dump the configuration.
if finalConfig is not None:  # have a finalConfig. If so save it. We could not have it if dry_run or read_only set.
    finalConfig.save(final_JSON_file)
    if monitor:
        finalConfig.plot(monitorFile=monitor_file)  # plot "std plot"