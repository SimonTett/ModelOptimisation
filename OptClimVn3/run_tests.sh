#!/bin/env bash
# run all tests. First run . setup to set up search paths.
export PYTHONPATH=$PYTHONPATH:$OPTCLIMTOP/OptClimVn3/test_OptClimVN3:$OPTCLIMTOP/OptClimVn3/Models/test_Models
for f in ${OPTCLIMTOP}/OptClimVn3/Models/test_Models/test_*.py  ${OPTCLIMTOP}OptClimVn3/test_OptClimVN3/test_*.py ${OPTCLIMTOP}OptClimVn3/support/test_support/test_*.py
do 
    echo $f 
    python -m unittest $f 
    # did test work?
    status=$?
    if [[ "$status" -ne 0 ]] ; then # no
	echo "Test case $f failed. Fix and rerun by:" 
	echo "python -m unittest $f"
	exit # exist. User please fix!
    fi
	
    echo "=======" 
done


echo "All tests done"
echo "====================================="
