#!/usr/bin/env python
"""
Dummy post processing script. Loads a load of modules to test that works ok and then copies output from fakemodel.py
"""

import numpy  # numpy
import datetime # get datetime module
import iris # Met Office IRIS library
from iris.time import PartialDateTime
import iris.analysis.cartography
import json # get JSON library
import copy # allow copying of objects
import argparse # parse arguments
import os # OS support
import glob # glob support.
import shutil

parser=argparse.ArgumentParser(description="Dummy post processing script")
parser.add_argument("JSON_FILE",help="The Name of the JSON file")
parser.add_argument("OUTPUT",help="The Name of the output file")
parser.add_argument("-v","--verbose",help="Provide verbose output",action="count",default=0)
args=parser.parse_args() # and parse the arguments
# setup processing
json_file=open(args.JSON_FILE,'r') # open file for reading
json_info=json.load(json_file) # read the JSON file
json_file.close() # close file

verbose=args.verbose
output=args.OUTPUT
try:
    options=json_info['postProcess'] # get the options
except KeyError:
    raise Exception("Need to provide post_process_options in "+args.JSON_FILE)

input=os.path.join("A","modelRun.nc")
shutil.copyfile(input,output)
if verbose: print "Copied %s to %s"%(input,output)

