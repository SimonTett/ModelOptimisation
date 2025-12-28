#/bin/bash
# run optclim set_model_status.py to update model status.
# Which in turns does various actions.

echo "Optclim"
echo "Args: $*"
config=$1 ; shift

echo "hostname:"
hostname 

echo "pwd:"
pwd
echo "ROSE_SUITE_DIR: $ROSE_SUITE_DIR "
echo "ROSE_DATA: $ROSE_DATA "

echo "ls:"
ls

# comment out export as gives a lot. Might be worth making optional.
# (If verbose set) 
#echo "export:" 
#export

# Setup python environment -- PY_ENV_SETUP needs to be set in the rose suite:
. $PY_ENV_SETUP
echo "Virtual Env is: $VIRTUAL_ENV"

echo "ls -ltr $config:"
ls -ltr $config



set_model_status.py $config $* # actually set the status.
status=$?
if [[ ${status} -ne 0 ]]
   then
   echo "set_model_status.py $config $* failed with status $status"
   exit $status
fi
# python <optclim_script>.py "$@"
echo "ls after:"
ls -ltr

