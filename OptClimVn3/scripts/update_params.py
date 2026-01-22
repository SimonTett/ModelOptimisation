#!/usr/bin/env python
import json
import pathlib
import argparse
from runSubmit import runSubmit
import genericLib
from scripts.pp_script_test import output
from um_rose_files.ukesm_params import output_param

parser = argparse.ArgumentParser(description="Update parameters in an existing OptClimVn3 configuration file. "
                                             "Optionally copy the input configuration to a new file and output logcal_params and logical_obs to files.")
parser.add_argument('config_path', type=pathlib.Path, help='Path to the existing configuration file (.scfg)')
parser.add_argument('--parameters',  nargs='+',
                    help='List of parameters to update from model config')
parser.add_argument('--output', type=pathlib.Path,
                    help='Path to save the updated configuration file. If not provided, nothing will be saved.')
parser.add_argument('--output_obs', type=pathlib.Path,
                    help='Path to save the updated observations file. If not provided, observations will not be saved.')
parser.add_argument('--output_params', type=pathlib.Path,
                    help='Path to save the updated parameters file. If not provided, parameters will not be saved.')
parser.add_argument('--eval_db_path', type=pathlib.Path,help='Path to evaluation database. Only generated if output_obs and output_params are specified.')
parser.add_argument('--log', help="Logging level", default="WARNING",choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])


args = parser.parse_args()
if args.eval_db_path and not (args.output_obs and args.output_params):
    raise ValueError("If --eval_db_path is specified, both --output_obs and --output_params must also be specified.")

my_logger = genericLib.setup_logging(args.log)
my_logger.debug(f"Output path is: {args.output}")
my_logger.debug(f"Parameters are: {args.parameters}")


# Load existing configuration
config = runSubmit.load(args.config_path)
if config is None:
    raise  ValueError(f"Failed to load configuration from {args.config_path}")





config.update_params(args.parameters) # should not write anything out yet.

# write out the requested files
if args.output_obs: # want to write out  observations?
    config.logical_obs().to_csv(args.output_obs)
    my_logger.debug(f"Observations saved to {args.output_obs}")

params = config.logical_params()
if args.output_params: # Want to write out  parameters ?
    params.to_csv(args.output_params)
    my_logger.debug(f"Saved parameters to {args.output_params}")
if args.output: # Want to write out  modified config?
    new_config = config.copyConfig(args.output)  # copy modified config.
    my_logger.info(f"Updated configuration saved to {args.output}")
if args.eval_db_path:
    eval_config=dict(
        parameters=args.output_params.to_posix(),
        simulated_observations=args.output_obs.to_posix(),
        start_index=int(params.iloc[-1].name)
    )
    eval_config = {"evaluation_database":eval_config} # wrap it in evaluation_database
    with open(args.eval_db_path, 'w') as f:
        json.dump(eval_config,f)
    my_logger.info(f"Saved evaluation database to {args.eval_db_path}")

