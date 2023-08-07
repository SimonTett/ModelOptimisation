#!/bin/env python
#  script to set model status to defined status by running appropriate function.
#  These functions may have other effects beyond setting the status. This script should be called
# from the running model. Min necessary is to have the model say when it is completed.
# Nice is for it to say when it starts running and if it fails.
import argparse
import logging
import os
import pathlib
from Model import Model


allowed_keys = set(Model.status_info.keys()) - {'SUBMITTED'}
# this script does not handle submission of the model as that is more complex. See SubmitStudy for that.
parser = argparse.ArgumentParser(description="""
    Set model status to something. This can have side effects depending on the status. 
    Example usage: set_model_status pth_to_config COMPLETED
    """)
parser.add_argument("config",type=str,help='path for model config')
parser.add_argument("status", type=str, help="What to set model status to", choices=allowed_keys)
parser.add_argument("-v", "--verbose", action="count", default=0,
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
config_path = Model.expand(args.config) #
logging.info(f"config_path = {config_path}")
status = args.status
logging.info(f"Status = {status}")
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

