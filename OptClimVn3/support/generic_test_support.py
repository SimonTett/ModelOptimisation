# stuff for generic test support.
# sets up the OPTCLIMTOP environment variable.
# and for working out what engine we should be using
import pathlib
import os
import engine
import subprocess
raise RuntimeError("This module is not ready for use yet.")
# set up OPTCLIMTOP for tests.
here = pathlib.Path(__file__).parent
os.environ['OPTCLIMTOP'] = str(here.parent.parent) # two levels up.

