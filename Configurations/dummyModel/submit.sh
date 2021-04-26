#!/usr/bin/env bash
# dummy submit script
echo "running dummy model"
dir=$(dirname $0)
cd $dir
echo "CWD is $PWD"
. ../setupPy.sh # make sure we have a python env!
export PATH=$PWD:$PATH # so we run things from the model dir first.
# need to copy json configuration too... That's done by the main python code
qsub -cwd -V script.sh # submit the script that actually works...
