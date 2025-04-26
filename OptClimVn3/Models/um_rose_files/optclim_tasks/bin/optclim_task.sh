#/bin/bash

echo "Optclim"
echo "Args: $*"

echo "hostname:"
hostname 

echo "pwd:"
pwd

echo "ls:"
ls

echo "export:"
export

# Setup python environment -- PY_ENV_SETUP needs to be set in the rose suite:
. $PY_ENV_SETUP

# TODO: run optclim
# python <optclim_script>.py "$@"
