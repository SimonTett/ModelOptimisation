#!/bin/env python
#  script to set model status to defined status by running appropriate function.
#  These functions may have other effects beyond setting the status. This script should be called
# from the running model. Min necessary is to have the model say when it is completed.
# Nice is for it to say when it starts running and if it fails.
import argparse
import logging
import os
import pathlib
import generic_json
from  Model import Model # need some generic way of importing everything in Models
from Models.HadCM3 import HadCM3


allowed_keys = set(Model.status_info.keys()) - {'SUBMITTED'}
# this script does not handle submission of the model as that is more complex. See SubmitStudy for that.
parser = argparse.ArgumentParser(description="""
    Set model status to something. This can have side effects depending on the status. 
    Example usage: set_model_status COMPLETED
    This script gets the path to the model config from:
    --config_path
    os.environ('MODEL_CONFIG_PATH') 
    or reading (in the current directory) MODEL_CONFIG_PATH.json
    """)
parser.add_argument("--config_path",type=str,help='path for model config',default=None)
parser.add_argument("status", type=str, help="What to set model status to", choices=allowed_keys)
parser.add_argument("-v", "--verbose", action="count", default=None,
                    help="Be more verbose. Level one gives logging.INFO and level 2 gives logging.DEBUG")
args = parser.parse_args()
# deal with verbosity
if args.verbose == 1:
    logging.basicConfig(level=logging.INFO,force=True)
elif args.verbose == 2:
    logging.basicConfig(level=logging.DEBUG,force=True)
else:
    pass

for k,v in vars(args).items():
    logging.debug(f"arg.{k}={v}")
for k,v in os.environ.items():
    logging.debug(f"${k} = {v}")

logging.debug(f"Path is {pathlib.Path.cwd()}")
# work out where config lives.
config_path = args.config_path # first try arguments
if config_path is None: # not provided
    logging.debug("No config_path provided. Trying env")
    try:
        config_path=pathlib.Path(os.environ['OPTCLIM_MODEL_PATH']) # get from env.
    except KeyError: # not found in env
        file = "OPTCLIM_MODEL_PATH.json"
        logging.debug(f"No OPTCLIM_MODEL_PATH in env. Trying to read {file}")
        with open(file) as fp: # read json file and extract config_path
            config_path_dir = generic_json.load(fp)
        config_path = config_path_dir['config_path']
else:
    config_path = Model.expand(args.config_path)
status = args.status

# verify that dir_path exists and is a directory
if not (config_path.exists()):
    raise ValueError(f"config_path {config_path} does not exist.")
model = Model.load_model(config_path)
# read model in. type of model gets worked out through saved configuration
# and set its status. That can do lots of things. See Model methods.
# there is enough logging in load_model to report what is being done!

if status == 'INSTANTIATED':
    model.instantiate()
elif status == 'RUNNING':
    model.running()
elif status == 'FAILED':
    model.set_failed()
elif status == 'SUCCEEDED':
    model.succeeded()
elif status == 'PROCESSED':
    model.process()
else:
    raise ValueError(f"Status {args.status} unknown")

