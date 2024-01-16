# DFOLS appears to be non-deterministic when running with parallel initialisation
# this code is to actually run the code to produce config files.
import copy
import pathlib
import unittest
import subprocess
import tempfile
from Model import Model
import platform
import shutil

import copy

import StudyConfig
from runSubmit import runSubmit # so we can test if we have one!
script_dir = Model.expand("$OPTCLIMTOP/OptClimVn3/scripts")
# std config first
config_pth = Model.expand("$OPTCLIMTOP/OptClimVn3/configurations/dfols14param_opt3.json")
config = StudyConfig.readConfig(config_pth)
# now modify the config.
config_good=config.copy()
file_good = filename=pathlib.Path("mod_dfols14param_opt3.json")
config_good.name("dfols_mod")
opt =config_good.optimise()
opt['dfols']["raise_error"]=True # force raise error. Parallelism does not matter
#opt['dfols']['namedSettings']['init.run_in_parallel']=False
#opt['dfols']['namedSettings']['init.random_initial_directions']=False
config_good.optimise(**opt) # overwrite it!
config_good.save(verbose=True,filename=file_good)
std_dir = pathlib.Path("std_config")
good_dir=pathlib.Path('fix_config')
if std_dir.exists():
    shutil.rmtree(std_dir,ignore_errors=True)
if good_dir.exists():
    shutil.rmtree(good_dir,ignore_errors=True)
if platform.system() == 'Windows':
    std_cmd=['python']
    good_cmd=['python']
else:
    std_cmd=[]
    good_cmd = []


std_cmd += [str(script_dir/'runAlgorithm.py'),str(config_pth), "-t","-d",str(std_dir)]
good_cmd += [str(script_dir/'runAlgorithm.py'),str(file_good), "-t","-d",str(good_dir)]
print("Running std config. Will take a while")
res = subprocess.run(std_cmd,capture_output=True,text=True)
print("stdout", res.stdout)
print("stderr", res.stderr)
res.check_returncode()

config_pth = std_dir/(config.name() + ".scfg")
std_sconfig = runSubmit.load(config_pth)


# now run with the fixed config

print("Running good config. Will take a while")
res = subprocess.run(good_cmd,capture_output=True,text=True)
print("stdout", res.stdout)
print("stderr", res.stderr)
res.check_returncode()

config_pth = good_dir/(config_good.name() + ".scfg")
good_sconfig = runSubmit.load(config_pth)
for num,(std,good) in enumerate(zip(std_sconfig.model_index.keys(),good_sconfig.model_index.keys())):
    if std != good:
        print("diff at ",num)
        first_delta = num
        break

# lets work out the iter size for the std case
for itc,it in enumerate(std_sconfig.iterations()):
    if std in [i.key() for i in it]:
        print(itc,len(it))


