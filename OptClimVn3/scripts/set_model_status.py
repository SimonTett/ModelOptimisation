#!/bin/env python
#  script to set model status to defined status by running appropriate function. These functions may have other effects beyond setting the status.
import argparse
import logging
import pathlib

from  Model import Model

allowed_keys = set(Model.status_info.keys()) - set(['SUBMITTED'])
# this script does not handle submission of the model as that is more complex.
parser = argparse.ArgumentParser(description="""
    Set model status to something. This can have side effects depending on the status. 
    Example usage: set_model_status COMPLETED""")
parser.add_argument("config_path",type='str',help='path for model config')
parser.add_argument("status", type=str, help="What to set model status to", choices=allowed_keys)
parser.add_argument("-v", "--verbose", action="count", default=None,
                    help="Be more verbose. Level one gives logging.INFO and level 2 gives logging.DEBUG")
args = parser.parse_args()

config_path = pathlib.Path(args.config_path)


if args.verbose == 1:
    logging.basicConfig(level=logging.INFO)

elif args.verbose == 2:
    logging.basicConfig(level=logging.DEBUG)
else:
    pass

status = args.status

# verify that dir_path exists and is a directory
if not (config_path.exists()):
    raise ValueError(f"config_path {config_path} does not exist.")


model = Model.load_model(config_path)
# read model in. type of model gets worked out through saved configuration.

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
