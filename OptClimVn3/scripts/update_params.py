#!/usr/bin/env python
import json
import pathlib
import argparse
from runSubmit import runSubmit
import genericLib
from StudyConfig import readConfig
import datetime
import sys

# TODO refactor this into code that just supports DFOLS. And review how DFOLS code handles it.
#  Updating params while running not to be supported. Best approach there is to start a new case and import old model runs.
parser = argparse.ArgumentParser(description="Update parameters in an existing OptClimVn3 configuration file. "
                                             "Optionally copy the input configuration to a new file and output logcal_params and logical_obs to files.")
parser.add_argument('config_path', type=genericLib.expand, help='Path to the existing configuration file (.scfg)')
group = parser.add_mutually_exclusive_group(required=False) # support getting parameters defined in two different ways
group.add_argument('--parameters',  nargs='+',
                    help='List of parameters to update from model config')

group.add_argument('--study_config_path', type=genericLib.expand, help='StudyConfig to use to define parameters. If provided will be used to update config.')
parser.add_argument('--output', type=genericLib.expand,
                    help='Path to save the updated configuration file. If not provided, nothing will be saved. Parent dir will be used to save modified model files')
parser.add_argument('--output_obs', type=genericLib.expand,
                    help='Path to save the updated observations file. If not provided, observations will not be saved.')
parser.add_argument('--output_params', type=genericLib.expand,
                    help='Path to save the updated parameters file. If not provided, parameters will not be saved.')
parser.add_argument('--reindex',action=argparse.BooleanOptionalAction,
                    help='Reindex params by obs index. Needed for legacy non determinism')
parser.add_argument('--eval_db_path', type=genericLib.expand,help='Path to evaluation database. Only generated if output_obs and output_params are specified.')
parser.add_argument('--log', help="Logging level", default="WARNING",choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])


args = parser.parse_args()
if args.eval_db_path and not (args.output_obs and args.output_params):
    raise ValueError("If --eval_db_path is specified, both --output_obs and --output_params must also be specified.")


my_logger = genericLib.setup_logging(args.log)
my_logger.debug(f"Output path is: {args.output}")

# check that output_args.parent does not exist!
if args.output and (args.output.parent.exists()):
    raise FileExistsError(f"{args.output.parent} exists. ")

# Load existing configuration

study_config = None
config = runSubmit.load(args.config_path)
if config is None:
    raise  ValueError(f"Failed to load configuration from {args.config_path}")

# setup parameters
if args.parameters:
    parameters=args.parameters
    study_config = None
elif args.study_config_path:
    study_config = readConfig(args.study_config_path)
    parameters = study_config.paramNames()
else:
    parameters=[] # no parameters to update

if parameters: # got some parameters
    my_logger.debug(f"Parameters to update are {' '.join(parameters)}")

    config.update_params(parameters) # should not write anything out yet.

# write out the requested files
if args.output_obs: # want to write out  observations?
    config.simulated_observations().to_csv(args.output_obs) # Want scaled obs.
    my_logger.debug(f"Observations saved to {args.output_obs}")

params = config.logical_params()
if args.reindex: # reindex params by obs keys. Bit of a hack to cope with non-deterministic cases
    indx =  config.simulated_observations(scale=False).index
    params = params.reindex(indx)
    my_logger.info(f"Reindex params -- now has shape {params.shape}")
if args.output_params: # Want to write out  parameters ?
    params.to_csv(args.output_params)
    my_logger.debug(f"Saved parameters to {args.output_params}")
if args.output: # Want to write out  modified config?
    if study_config is not None: # generate a new config with the study config used to generate the params
        # Approach here is to create a new runSubmit object using the study_config provided.
        # Then take the updated models (with new params), and copy them -- updating paths as we do so.
        # Finally use read_model_configs to include the modified models in the config.
        my_logger.info(f"Creating new config from {study_config.fileName()}")
        rootDir=args.output.parent 
        rootDir.mkdir(parents= True) # make root dir (and all parents). Will fail if directory exists.

        new_config = runSubmit(study_config, rootDir=rootDir,
                                         config_path=args.output,name=args.output.name  )
        new_files=[]
        for model in config.model_index.values(): # get the models and copy them into the new directory
            new_dir = rootDir/(model.model_dir.relative_to(config.rootDir) )# new directory for model.
            m =  model.copyConfig(new_dir, update_paths=True)
            new_files.append(m.config_path)
        my_logger.info(f"Copied {len(new_files)} models")
        new_config.read_model_configs(new_files) # and get the models configs
        new_config.update_history(f"Copied {len(new_files)} models from {config.rootDir} with params updated.")
        new_config.dump_config(dump_models=False) # no need to dump the models as already done so.
        
    else:
        new_config = config.copyConfig(args.output.parent)  # copy modified config.
        my_logger.info(f"Updated configuration saved to {args.output.parent}")
        # move the new confile file to the new name
        if config.config_path.name != args.output.name: # only move if different name
            new_config.config_path.rename(args.output)
            my_logger.info(f"Renamed configuration file to {args.output}")

if args.eval_db_path:
    comment = f"Generated using {__file__} at {datetime.datetime.now()} with cmd line args {' '.join(sys.argv)}"
    my_logger.debug(comment)
    eval_config=dict(
        parameters=args.output_params.as_posix(),
        simulated_observations=args.output_obs.as_posix(),
        start_index=params.iloc[-1].name,
        _comment=comment
    )
    eval_config = {"evaluation_database":eval_config} # wrap it in evaluation_database
    with open(args.eval_db_path, 'w') as f:
        json.dump(eval_config,f,indent=2)
    my_logger.info(f"Saved evaluation database to {args.eval_db_path}")

