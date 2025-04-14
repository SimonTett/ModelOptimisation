#/bin/bash

echo "Optclim"
echo "Args: $@"

echo "hostname:"
hostname 

echo "pwd:"
pwd

echo "ls:"
ls

echo "export:"
export

# Activate python environment:
. $OPTCLIM_PY_ENV

# TODO: run optclim
# python <optclim_script>.py "$@"
