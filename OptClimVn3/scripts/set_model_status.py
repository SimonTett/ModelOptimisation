#!/bin/env python
#  script to set model status to defined status by running appropriate function. These functions may have other effects beyond setting the status.
import argparse
import logging
import pathlib

import Models

allowed_keys = set(Models.status_info.keys()) - set(['SUBMITTED'])
# this script does not handle submission of the model as that is more complex.
parser = argparse.ArgumentParser(description="""
    Set model status to something. This can have side effects depending on the status. 
    Example usage: set_model_status COMPLETED""")
parser.add_argument("status", type=str, help="What to set model status to", choices=allowed_keys)
parser.add_argument("-d", "--directory",
                    help="directory name where config lives. If not specified will use current working dir_path",
                    default=None, type=str)
parser.add_argument("-v", "--verbose", action="count", default=None,
                    help="Be more verbose. Level one gives logging.INFO and level 2 gives logging.DEBUG")
args = parser.parse_args()

dir_path = args.directory
if dir_path is None:
    dir_path = pathlib.Path().cwd()
else:
    dir_path = pathlib.Path(dir_path)

if args.verbose == 1:
    logging.basicConfig(level=logging.INFO)

elif args.verbose == 2:
    logging.basicConfig(level=logging.DEBUG)
else:
    pass

status = args.status

# verify that dir_path exists and is a directory
if not (dir_path.exists()):
    raise ValueError(f"dir_path {dir_path} does not exist.")
if not dir_path.is_dir():
    raise ValueError(f"dir_path {dir_path} is not a directory.")

model = Models.Model.Model.load_model(dir_path)
# read model in. type of model gets worked out through saved configuration.

if status == 'INSTANTIATED':
    model.instantiate()
elif status == 'RUNNING':
    model.running()
elif status == 'FAILED':
    model.failed()
elif status == 'SUCCEEDED':
    model.succeeded()
elif status == 'PROCESSED':
    model.process()
else:
    raise ValueError(f"Status {args.status} unknown")
