#!/usr/bin/env python

import pathlib
import argparse
from runSubmit import runSubmit
import genericLib



parser = argparse.ArgumentParser(description="Update parameters in an existing OptClimVn3 configuration file.")
parser.add_argument('config_path', type=pathlib.Path, help='Path to the existing configuration file (.scfg)')
parser.add_argument('--parameters',  nargs='+',
                    help='List of parameters to update. Will get values from config file',)
parser.add_argument('output', type=pathlib.Path,
                    help='Path to save the updated configuration file.')
parser.add_argument('--log', help="Logging level", default="WARNING",choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])

args = parser.parse_args()

my_logger = genericLib.setup_logging(args.log)
my_logger.debug(f"Output path is: {args.output}")
my_logger.debug(f"Parameters are: {args.parameters}")


# Load existing configuration
config = runSubmit.load(args.config_path)
if config is None:
    raise  ValueError(f"Failed to load configuration from {args.config_path}")


# Save updated configuration
output_direct = args.output
new_config = config.copyConfig(output_direct)
new_config.update_params(args.parameters)
new_config.dump_config(dump_models=True)
my_logger.info(f"Updated configuration saved to {output_direct}")

