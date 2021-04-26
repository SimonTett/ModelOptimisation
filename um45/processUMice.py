#!/usr/bin/env python
"""
Generate ice data for analysis.
"""
# TODO add some meta-data to output. Life short so have not done so.
# Thing to include are root dir, cachedirs, name of this code, data ran, input
# example run is  $OPTCLIMTOP/um45/processUM.py studies/coupJac/coupJac14p.json


import argparse  # parse arguments
import json
import os  # OS support
import pathlib

import ppSupport
import iris
import StudyConfig  # needed to parse json file.



def minMaxIceE(ts,year=slice(11,21),scale =1):
    """
    Compute min and max ice extent from ice_extent timeseries
    :param ts: iris timeseries of ice extent site_number =1 = NH; 2 = SH
    :param year slice object for year range wanted.
    :param scale -- scaling to apply to timeseries. Set to 1e12 if want results in 10^6 km^2
    :return:dict with NHmax, NHmin, SHmax, SHmin  -- ice extent in m^2
    """


    # work out iris.Constraint
    yrconstraint = iris.Constraint(year=lambda cell: year.start <= cell < year.stop)
    shortIce = ts.extract(yrconstraint)
    ann_cycle = shortIce.aggregated_by('month_number',iris.analysis.MEAN)/scale
    result={}
    for site,name in zip([1,2],['N','S']):
        tt = ann_cycle.extract(iris.Constraint(site_number=site))
        result[name+'mx']=float(tt.data.max())
        result[name + 'mn'] = float(tt.data.min())
    return result


parser = argparse.ArgumentParser(description=
                                 """Post process Unified Model to:
  1) Run mkphh on at least monthly  mean ocean   data
  2) Run ummonitor (via IDL) on monthly mean ocean data.
  Assume to be running in UM directory
  
  Example use is:
  
   processUMice.py input.json 
  
  which will run makepph & ummonitor on the A/[ao]py dirs.
  """
                                 )
parser.add_argument("JSON_FILE", help="The name of the JSON file. Use the postProcess block to define directories to process and output file ")
parser.add_argument("OUTPUT_FILE", help="The name of the output JSON file ")
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
#outputFile = rootDir / config.postProcessOutput()
outputFile = pathlib.Path(args.OUTPUT_FILE)
if outputFile.suffix != '.json':
    raise Exception("Make your output file be json.")

# get information from dirs for post processing.
dirConfig = config.getv('postProcess').get('dirs',{})
# make sure have opm with ts_ice_extent
opm = dirConfig.get('opm',[])
opm.append('ts_ice_extent')
dirConfig['opm']=opm
# and apy for 1.5m temp
apy = dirConfig.get('apy',[])
apy.append('ts_t15')
dirConfig['apy']=apy

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

# now processed all data get the monthly mean ice_extent_data and process it.
# file name easy to make as long as everything standard...
ice_file =  cacheroot / (rootDir.name+'.000001') / 'ts_ice_extent.pp'
ts = ppSupport.readPP(ice_file) # get the ice_extent T/S
yearRange = dirConfig.get('yearRange',[11,21])
year = slice(yearRange[0],yearRange[1])
ice = minMaxIceE(ts,year=year)
cacheInfo.update(ice)
# and compute the global mean temperature
temp_file = cacheroot / (rootDir.name+'.000100') / 'ts_t15.pp'
t15 = ppSupport.readPP(temp_file)
t15 = t15.extract(iris.Constraint(site_number=3,
                        year = lambda yr:  yearRange[0] <= yr < yearRange[1]))
cacheInfo.update(SAT=float(t15.data.mean()))
with open(outputFile, 'w') as outfile:
    json.dump(cacheInfo, outfile)

## end of routine.
