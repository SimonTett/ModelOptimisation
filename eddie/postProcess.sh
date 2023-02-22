#!/bin/bash
# Script to post process the data.
# takes 2 arguments
# dirListFile -- file containing the directories to process
# jsonFile -- the filename of the JSON configuration file.
for var in "$@" ; do
    echo "ARG: >>>$var<<<"
done
DirListFile=$1  # file containing list of directories.
JsonFile=$2  # json file
JsonFile=$(realpath $JsonFile)
echo "going to post process using  $JsonFile in $PWD from $DirListFile"
. /etc/profile.d/modules.sh
echo "Array ID $SGE_TASK_ID"
if [[ $SGE_TASK_ID -gt 0 ]] ; then
    if [[  ! -f $DirListFile ]] ;  then
       pwd
       echo "in wrong place?"
       echo "not found $DirListFile"
       exit 1
    fi
    # extract directory and name of output file.
    line=$(awk "NR==$SGE_TASK_ID" $DirListFile)
    # now split line on , into dir, ppScript and PostProcessOutput
    dir=$(echo $line | cut -f1 -d",")
    PostProcessScript=$(echo $line | cut -f2 -d",")
    PostProcessOutput=$(echo $line | cut -f3 -d",")
    cd $dir
    echo "going to run $PostProcessScript $JsonFile $PostProcessOutput in $PWD"
    $PostProcessScript $JsonFile $PostProcessOutput
else
    echo "SGE_TASK_ID <= 0 = $SGE_TASK_ID"
fi


