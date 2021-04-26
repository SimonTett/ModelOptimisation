#!/bin/bash
# Script to post process the data.
# takes 2 arguments
# dirListFile -- file containing the directories to process
# jsonFile -- the filename of the JSON configuration file.
echo "Args are $*"
for var in "$@" ; do
    echo ">>>$var<<<"
done
DirListFile=$1  # file containing list of directories.
JsonFile=$2  # json file
echo "DirListFile is $DirListFile"
echo "JsonFile is $JsonFile"
JsonFile=$(realpath $JsonFile)
echo "going to post process using  $JsonFile in $PWD"
echo pwd=$PWD
. /etc/profile.d/modules.sh
#. ./setupPy.sh ## run the setup script which sets up paths etc.
echo "Array ID "$SGE_TASK_ID
if [ $SGE_TASK_ID > 0 ] ; then
    if [  ! -f $dirListFile ] ;  then
       pwd
       echo in wrong place?
       echo not found $dirListFile
       exit 1
    else # extract directory and name of output file.
       line=$(awk "NR==$SGE_TASK_ID" $DirListFile)
       # now split line on , into dir, ppScript and PostProcessOutput
       dir=$(echo $line | cut -f1 -d",")
       PostProcessScript=$(echo $line | cut -f2 -d",")
       PostProcessOutput=$(echo $line | cut -f3 -d",")
       echo $line
       cd $dir
       echo "going to run $PostProcessScript $JsonFile $PostProcessOutput in $PWD"
       $PostProcessScript $JsonFile $PostProcessOutput
  fi
fi


