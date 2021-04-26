#!/usr/bin/env python
"""
Run mkpph & ummonitor on files.
Then work out where it has got to and save that information
"""
# TODO add some meta-data to output. Life short so have not done so.
# Thing to include are root dir, cachedirs, name of this code, data ran, input
# example run is  $OPTCLIMTOP/um45/processUM.py studies/coupJac/coupJac14p.json


import argparse  # parse arguments
import json
import os  # OS support
import pathlib

import ppSupport

import StudyConfig  # needed to parse json file.

maxYear = {}



parser = argparse.ArgumentParser(description=
                                 """Post process Unified Model to:
  1) Run mkphh on annual average atmospheric data
  2) Run ummonitor (via IDL) on annual average atmospheric data
  3) read in ts_sst.pp & ts_t15.pp from op? and ap? dirs 
    and work out max year
    -- NB this not really used. Subseqent calculations want timeseries. 
  Assume to be running in UM directory
  
  Example use is:
  
   processUM.py input.json 
  
  which will run makepph & ummonitor on the A/[ao]py dirs.
  """
                                 )
parser.add_argument("JSON_FILE", help="The Name of the JSON file. Use the postProcess block to define directories to process and output file ")
parser.add_argument("--dir", "-d",
                    help="Name of directory to run on. Cached ummonitor files are put here. Default cwd",
                    default=os.getcwd())

parser.add_argument("-s", "--skip", help='Skip data processing. Just read in data', action='store_true')
parser.add_argument("-v", "--verbose", help="Provide verbose output", action="count", default=0)
args = parser.parse_args()  # and parse the arguments
verbose = args.verbose

rootDir = pathlib.Path(args.dir)
experName = rootDir.name  # really should work out from pp data but life is short


cacheroot = rootDir / 'A'  # where we want to put the data

config = StudyConfig.readConfig(args.JSON_FILE)  # config.
outputFile = rootDir / config.postProcessOutput()
# dirs to process are 
dirConfig = config.getv('postProcess')['dirs']

cacheInfo = dict()
for mnDir, choices in dirConfig.items():
    dir = cacheroot / mnDir  # full dir path
    if args.skip:  # skip data processing
        continue
    if args.verbose:
        print("Processing data in %s" % mnDir)
    # check dir exists and if not skip
    if not dir.is_dir():
        continue 
    # process it
    
    (cachedir, maxYear) = ppSupport.ummonitorDir(dir, choices, verbose=verbose)

    cacheInfo[cachedir] = maxYear
    if verbose:
        print( f"Dir {cachedir} maxYr {maxYear:5.1f} ")

# now processed all data write information to JSON file.
with open(outputFile, 'w') as outfile:
    json.dump(cacheInfo, outfile)
