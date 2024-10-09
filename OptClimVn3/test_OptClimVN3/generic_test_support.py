# stuff for generic test support.
# Currently, this only sets up the OPTCLIMTOP environment variable.
import pathlib
import os

# set up OPTCLIMTOP for tests.
here = pathlib.Path(__file__).parent
os.environ['OPTCLIMTOP'] = str(here.parent.parent) # two levels up.