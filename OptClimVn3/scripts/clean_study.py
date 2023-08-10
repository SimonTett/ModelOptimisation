#!/usr/bin/env python3
"""
Clean or purge study
Command line args:
jsonFile: path to jsonFile defining configuration.

do clean_study.py -h to see what the remaining  command line arguments are.

"""
import os


import argparse  # parse command line arguments
import functools
import os
import sys
import pathlib
import numpy as np
from Models import *  # imports all models we know about. See Models/__init__.py Import this before anything else.
import optclim_exceptions
import runSubmit
import StudyConfig

import genericLib
import logging
import shutil


## main script

## set up command line args

parser = argparse.ArgumentParser(description="Delete study. Use --purge to remove whole directory")
parser.add_argument("-d", "--dir", help="path to root directory where model runs are")
parser.add_argument("jsonFile", help="json file that defines the study")
parser.add_argument("--purge", action='store_true',
                    help="purge the configuration by deleting the directory. Will ask if OK. Will run after Study & models deleted")
parser.add_argument("-v", "--verbose", action='count', default=0,
                    help="level of logging info level= 1 = info, level = 2 = debug ")
args = parser.parse_args()
verbose = args.verbose
jsonFile = pathlib.Path(os.path.expanduser(os.path.expandvars(args.jsonFile)))
purge = args.purge

if verbose == 1:
    level=logging.INFO

if verbose > 1:
    level=logging.DEBUG

if verbose: # turn on logging
    optclim_logger = logging.getLogger("OPTCLIM") # OPTCLIM root logger
    fmt = '%(levelname)s:%(name)s:%(funcName)s %(message)s'
    formatter = logging.Formatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.basicConfig(force=True,format=fmt,level=level)
    optclim_logger.addHandler(console_handler)
    optclim_logger.setLevel(level)

configData = StudyConfig.readConfig(filename=jsonFile)  # parse the jsonFile.


if args.dir is not None:
    rootDir = Model.expand(args.dir)  # directory defined so set rootDir
else:  # set rootDir to cwd/name
    rootDir = pathlib.Path.cwd() / configData.name()  # default path
 


logging.info("Running from config %s named %s" % (jsonFile, configData.name()))
config_path = rootDir / (configData.name() + ".scfg")


##############################################################
# Main block of code
#  Now we actually clean
##############################################################
logging.info(f"Reading status from {config_path}")
rSUBMIT = runSubmit.runSubmit.load_SubmitStudy(config_path)
if not isinstance(rSUBMIT, runSubmit.runSubmit):
    raise ValueError(f"Something wrong")

logging.info(f"Deleting existing config {rSUBMIT}")
rSUBMIT.delete()  # should clean dir, kill jobs etc.

if purge: # purging data? Do after deleting so as to clean up.
    result = input(f">>>Going to delete all in {rootDir}<<<. OK ? (yes if so): ") 
    if result.lower() in ['yes']:
        print(f"Deleting all files in {rootDir} and continuing")
        shutil.rmtree(rootDir, onerror=genericLib.errorRemoveReadonly)
    else:
        print(f"Nothing deleted.")
