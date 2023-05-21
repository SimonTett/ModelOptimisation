#!/bin/bash
# Script to post process the data.
# takes 1 arguments
# configListFile -- list of Model configurations to process.
configListFile=$1  # file containing list of directories.
echo $configListFile
# check it exists.
if [[  ! -f $configListFile ]]
  then
     echo "in $PWD and not found $configListFile"
     exit 1
fi
#. /etc/profile.d/modules.sh
cmd=set_model_status
if [[ "$SGE_TASK_ID" ]]; then
  TASK_ID=$SGE_TASK_ID
  echo "Using SGE. "
elif [[ "$SLURM_ARRAY_TASK_ID" ]];then
  TASK_ID=$SLURM_ARRAY_TASK
  echo "Using SLURM. "
elif [[ "$TASK_ID" ]]; then
  echo "You must be faking it"
else
  echo "Not running in a known job control system"
  exit 1
fi

echo "Array ID = $TASK_ID"
if [[ $TASK_ID -gt 0 ]]
  then
      # extract file and name of output file.
      configFile=$(awk "NR==$TASK_ID" $configListFile)
      echo "going to run $cmd in $PWD"
      $cmd $configFile processed # will run the post processing.
else
    echo "TASK_ID <= 0 = $TASK_ID"
    exit 1
fi