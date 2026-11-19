#/bin/env bash
# run all tests. First run . setup to set up search paths.
# check that PROJECT_CODE is set. Needed for some tests
if [[ -z ${PROJECT_CODE} ]] ; then
    echo "Set PROJECT_CODE variable to what ever project you want to use before running this script"
    exit 0
fi
export PYTHONPATH=$PYTHONPATH:$OPTCLIMTOP/OptClimVn3/test_OptClimVN3:$OPTCLIMTOP/OptClimVn3/Models/test_Models
for f in OptClimVn3/Models/test_Models/test_*.py  OptClimVn3/test_OptClimVN3/test_*.py OptClimVn3/support/test_support/test_*.py 
do 
    echo $f 
    python -m unittest $f 
    # did test work?
    status=$?
    if [[ "$status" -ne 0 ]] ; then # no
	echo "Test case $f failed. Fix and rerun by:" 
	echo "python -m unittest $f"
	exit # exit. User please fix!
    fi
	
    echo "=======" 
done


echo "All tests done"
echo "====================================="
