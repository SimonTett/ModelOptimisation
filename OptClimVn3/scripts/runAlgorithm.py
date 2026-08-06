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

import logging
import argparse  # parse command line arguments
import functools
import os

import pathlib
import numpy as np
import shutil
import StudyConfig
import genericLib
import sys
import engine
# do minimum startup stuff. Really so can have logging

## main script

## set up command line args
expected_env_vars = ['OPTCLIM_ROOT_DIR', 'OPTCLIM_LOG_DIR', 'OPTCLIM_JOB_ID']
parser = argparse.ArgumentParser(description=f"Run study and provides the follow env variables: {' '.join(expected_env_vars)}" ,allow_abbrev=False)
parser.add_argument("-d", "--dir", help="path to root directory where model runs will be created")
parser.add_argument("jsonFile", help="json file that defines the study")
parser.add_argument("--delete", action='store_true',
                    help="Delete the configuration. Will ask of OK. If not will exit with status 1")
parser.add_argument("--purge", action='store_true',
                    help="purge the configuration by deleting the directory. Will ask if OK. If not OK will exit with status 1")
parser.add_argument("-v", "--verbose", action='count', default=0,
                    help="level of logging info level= 1 = info, level = 2 = debug ")
#parser.add_argument("--clean",help="Do not do anything but --delete and and --purge (if set)",action='store_true')
parser.add_argument("--dryrun", action='store_true',
                    help="if set do not submit any jobs but do instantiate models. Good for testing")
parser.add_argument("--readonly", action='store_true', help="read data but do not instantiate or submit jobs.")
parser.add_argument("-t", "--test", action='store_true',
                    help='If set run fake function rather than submitting models.')
parser.add_argument("--update_config", action='store_true',
                    help="If set update *existing* configuration from configuration given. Be careful when doing this.")
parser.add_argument("-m", "--monitor", action='store_true', help='Producing monitoring plot after running')
parser.add_argument('--kill',action='store_true',help='Kill all jobs in the study. ')
parser.add_argument('--process',action='store_true',help='Process all models that are SUCCEEDED and have no jobs. Then continue.')
parser.add_argument('--model_pattern',type=str,default=None,
                    help='glob pattern relative to the rootDir to load models when config is generated for first time. A good choice is "*/*.mcfg". Not tested. ')
parser.add_argument('--set_local_root_dir',action=argparse.BooleanOptionalAction,default=True,
                    help='If true (default) then if run_info.local_root_dir is not set it to root_dir. ')
fail_help_str = """Behaviour for models that failed. Choices are:
                fail (default), 
                continue (continue run with no changes)), 
                perturb (perturb run, restart run), 
                perturbc (perturb run, continue run),
                delete (delete model).
            These cases will be submitted (unless delete or fail are choice)"""
parser.add_argument("--fail", help=fail_help_str,
                    default='fail',
                    choices=['fail', 'continue', 'perturb', 'perturbc', 'delete'])

parser.add_argument("--archive",action='store_true',help="Archive configuration as tar file.")

parser.add_argument("--guess_fail", action='store_true',
                    help="If set then use guess_fail to see if Running models have failed and set them failed.")
parser.add_argument("--stop", action='store_true',
                    help="Stop the algorithm which is algorithm specific and might require more models being ran.")
args = parser.parse_args()
verbose = args.verbose
dry_run = args.dryrun
read_only = args.readonly
testRun = args.test
jsonFile = pathlib.Path(os.path.expanduser(os.path.expandvars(args.jsonFile)))
delete = args.delete
monitor = args.monitor
fail = args.fail
purge = args.purge
guess_fail = args.guess_fail
update_config = args.update_config
archive = args.archive
configData = StudyConfig.readConfig(filename=jsonFile)  # parse the jsonFile.
if args.dir is not None:
    rootDir = genericLib.expand(args.dir)  # directory defined so set rootDir
else:  # set rootDir to cwd/name
    rootDir = pathlib.Path.cwd() / configData.name()  # default path

output_dir = rootDir / 'jobOutput'
output_dir.mkdir(parents=True, exist_ok=True)  # make sure output dir (and in current implementation rootDir exists.)
# setup env vars for use in logging etc
genericLib.setup_config_env(rootDir,output_dir)



# check we have what is expected
for var in expected_env_vars:
    if var not in os.environ:
        raise ValueError(f"Expected env variable {var} not set. Fix runAlgorithm. See expected_env_vars in runAlgorithm.py for what is expected. ")



# logging stuff.
level = None
if verbose == 1:
    level=logging.INFO
if verbose > 1:
    level=logging.DEBUG

## set up the default logging.
fmt = '%(levelname)s:%(name)s:%(funcName)s: %(message)s'
logging.basicConfig(
    level=level,
    stream=sys.stdout,
    format=fmt,
    force=True
)

my_logger = genericLib.setup_logging(
    level=level,
    log_config=configData.logging_config()
)

# Now check the config is OK
configData.check()


# Import rest of stuff. Have logging on so we can see various auto-stuff in the
# class definitions

from Model import  Model # root type for all Models.
import optclim_exceptions
import runSubmit
import archive_study


if args.set_local_root_dir and (configData.run_info().get('local_root_dir') is None):
    configData.run_info()['local_root_dir'] = str(rootDir)
    my_logger.info(f"Set run_info.local_root_dir to {rootDir} ")
 

if purge: # purging data? Do early to minimize the amount of output user sees before this.
    result = input(f">>>Going to delete all in {rootDir}<<<. OK ? (yes if so): ") 
    if result.lower() in ['yes']:
        print(f"Deleting all files in {rootDir} and continuing")
        shutil.rmtree(rootDir, onerror=genericLib.errorRemoveReadonly)
    else:
        print(f"Nothing deleted.")
        sys.exit(1)

my_logger.info(f"Known models are {', '.join(Model.known_models())}")
my_logger.info("Running from config %s named %s" % (jsonFile, configData.name()))



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

rSUBMIT = None  # set it to None
if config_path.exists():  # config file exists. Read it in.
    my_logger.info(f"Reading status from {config_path}")
    rSUBMIT = runSubmit.runSubmit.load(config_path)

    if not isinstance(rSUBMIT, runSubmit.runSubmit):
        raise ValueError(f"Something wrong")
    if update_config:
        rSUBMIT.update_config(configData)
        # this will overwrite the existing configuration and change anything derived in it.
        # use with care
        my_logger.warning("Updated configuration. Your configuration might not be consistent.")

    if args.kill:
        my_logger.info(f"Killing all jobs in {rSUBMIT}")
        rSUBMIT.kill()

    if args.process:
        my_logger.info(f"Processing all models in {rSUBMIT} that are SUCCEEDED and have no jobs.")
        rSUBMIT.process()
        cmd = rSUBMIT.engine.kill_job(rSUBMIT.next_iter_jids[1]) # kill the next iter job.
        rSUBMIT.run_cmd(cmd)
        my_logger.info(f"Killing next iteration job with cmd: {cmd}")

    if delete:  # delete the config
        result = input(f">>>Going to delete existing configs in {rootDir}<<<. OK ? (yes if so): ") 
        if result.lower() in ['yes']:
            print(f"Deleting all configs in {rootDir} and continuing")
            my_logger.info(f"Deleting existing config {rSUBMIT}")
            rSUBMIT.delete()  # should clean dir and kill runs
            rSUBMIT = None  # remove it.
        else:
            print(f"Nothing deleted.")
            sys.exit(1) # exit.




if rSUBMIT is None:  # no configuration exists. So create it.
    # We can get here either because config_path does not exist or we deleted the config.
    args_not_for_restart = ['--delete','--purge','--update','--update_config','--kill','--process'] # logical flags
    # Arguments to be removed from the restart cmd
    restartCMD = [arg for arg in sys.argv if arg not in args_not_for_restart]  # generate restart cmd.
    # Remove --model_pattern's value if it's in the list
    if '--model_pattern' in restartCMD:
        idx = restartCMD.index('--model_pattern')
        del restartCMD[idx:idx + 2]  # remove the flag and its value
    my_logger.info(f"restartCMD is {restartCMD}")
    rSUBMIT = runSubmit.runSubmit(configData, rootDir=rootDir, config_path=config_path,next_iter_cmd=restartCMD)
    if args.model_pattern is not None: # we have a model pattern to load models from.
        my_logger.warning(f"Loading models from {args.model_pattern} in {config_path.parent}. Not yet tested")
        files = sorted(list(config_path.parent.glob(args.model_pattern)))
        if len(files) == 0:
            raise ValueError(f"Failed to find any files with {config_path.parent}.glob({args.model_pattern})")

        rSUBMIT.read_model_configs(files)
                
        my_logger.info(f"Loaded models in {files} ")
    my_logger.debug(f"Created new runSubmit {rSUBMIT}")
else:
    my_logger.debug(f"Using existing runSubmit {rSUBMIT}")



# We might  have runs to do so check that and run them if so.
if not (dry_run or read_only):  # not dry running or read only.
    # so first deal with failed models and then models that are already instantiated and so need running.
    # This happens if not all models that were instantiated were submitted.
    if guess_fail: # guess if models have failed. See Model.guess_failed to see how that is done.
        models_guess_failed = rSUBMIT.guess_failed()
    # test for RUNNING models. If any fail.
    running_models = rSUBMIT.running_models()
    if len(running_models) > 0:
        raise ValueError(f"{rSUBMIT} has {len(running_models)} running. Try --guess_fail if those have failed. Otherwise wait...")
    failed_models = rSUBMIT.failed_models()
    if len(failed_models):  # Some runs failed. Use fail to decide what to do
        my_logger.info(f"{len(failed_models)} models failed. {rSUBMIT}")
        if fail == 'fail':
            raise ValueError(f"{rSUBMIT} has FAILED models. Try --fail option: \n"+fail_help_str)
        for model in failed_models:
            my_logger.debug(f"Dealing with fail = {fail} for model {model}")
            if fail in ['perturb', 'perturbc']:
                model.perturb()  # perturb  model
            if fail in ['perturbc', 'continue']:  # model to continue
                model.status = 'CONTINUE'
            if fail == 'delete':  # delete model
                rSUBMIT.delete_model(model)

# check status is only PROCESSED or INSTANTIATED.
status = rSUBMIT.status()
if any(s not in ['PROCESSED','INSTANTIATED'] for s in status):
    raise ValueError(f"Have unexpected status rSUBMIT:{rSUBMIT}")

algorithmName = configData.optimise()['algorithm'].upper()
non_determinisitic = configData.optimise().get('nondeterministic', 'fail')
my_logger.debug(f"Algorithm is {algorithmName} and non_deterministic is {non_determinisitic}")
if algorithmName in ['RUNOPTIMISED', 'JACOBIAN']:
    wantCost = False
else:
    wantCost = True
# want to save_final_config?
save_final_config = True
if algorithmName in ['DFOLS']:
    save_final_config = False # not for DFOLS as it saves it

finalConfig = None  # so we have something!
with rSUBMIT.lock(timeout=30) as lock: # 30 second timeout.
    # TODO Risk of stale file but really want to refactor runAlgorithm to seperate out create and run.
    while True:  # loop indefinitely so can have fake_fn. This really to test code/algorithm.
        try:  # run an algorithm iteration.
            np.random.seed(123456)  # init RNG though probably should go to the runXXX methods.
            # reset logical information
            rSUBMIT.reset_logical_info()
            if algorithmName == 'DFOLS':
                finalConfig = rSUBMIT.runDFOLS(scale=True,stop=args.stop)
            elif algorithmName == 'PYSOT':
                # pySOT -- probably won't work without some work.
                finalConfig = rSUBMIT.runPYSOT(scale=True,stop=args.stop)
            elif algorithmName == 'GAUSSNEWTON':
                finalConfig = rSUBMIT.runGaussNewton(scale=True,stop=args.stop)
            elif algorithmName == 'JACOBIAN':
                # compute the Jacobian.
                finalConfig = rSUBMIT.runJacobian(stop=args.stop)
            elif algorithmName == 'RUNOPTIMISED':  # run optimised case through configuration in JSON file.
                finalConfig = rSUBMIT.runOptimized(stop=args.stop)
            elif algorithmName == 'RUN_PARAMS':  # run params.
                finalConfig = rSUBMIT.run_params(scale=True,stop=args.stop)
            else:
                raise ValueError(f"Don't know what to do with Algorithm: {algorithmName}")
            rSUBMIT.check_deterministic(error=non_determinisitic)
            break  # we have finished running algorithm so can exit and go to final clear up.
        except optclim_exceptions.submitModel:  # error which triggers need to instantiate and run more models.
            rSUBMIT.check_deterministic(error=non_determinisitic) # check are still deterministic.
            if read_only:
                my_logger.info(f"read_only -- exiting")
                break  # exit the loop -- we are done as in read_only mode.
            iter_count = rSUBMIT.instantiate()  # instantiate all cases that need instantiation.
            # This also generates iteration information.
            my_logger.info(f"Instantiated {rSUBMIT}")
            if dry_run:  # nothing gets submitted or faked. So exit
                my_logger.info(f"dry_run -- exiting")
                break
            if args.stop: # stopping. Raise error if any runs to be submitted. This could change in future.
                if iter_count > 0:
                    raise ValueError(f"Stopping but Instantiating  {iter_count} cases")
                my_logger.info('Stopping')
                break # exit loop
            nModels = rSUBMIT.submit_all_models(fake_fn=fakeFn)  # this also saves the config.
            finalConfig = rSUBMIT.runConfig(scale=True, add_cost=wantCost)  # generate final configuration
            my_logger.info(f"On iteration {iter_count} submitted {nModels} models")
            try:
                my_logger.info(f"Last cost is {float(finalConfig.cost().iloc[-1])}")
            except IndexError:  # no cost.
                pass
            if fakeFn is None:  # no fake fn so time to exit. This is "normal" behaviour.
                my_logger.info("Exiting")
                break  # exit the run forever loop as no more runs should be submitted on this go.
            else:  # reload the configuration (and all models).
                # This necessary as writing out/reading in changes (slightly) the floating point value of some values
                # which in turn changes the way the algorithms behave.
                rSUBMIT = runSubmit.runSubmit.load(config_path)

        # end of try/except.

    # Deal with final stuff
    rSUBMIT.dump_config(dump_models=True)  # dump the configuration & all models

    if finalConfig is not None and save_final_config:  # have a finalConfig. If so save it. We could not have it if dry_run or read_only set.
        finalConfig.save(final_JSON_file)

    if monitor:
        rSUBMIT.plot(fname=monitor_file)  # plot "std plot"
    if archive:
        archive = archive_study.archive_study()
        archive.archive(rSUBMIT,
                        extra_paths=[final_JSON_file.relative_to(rootDir),
                                     monitor_file.relative_to(rootDir)])
