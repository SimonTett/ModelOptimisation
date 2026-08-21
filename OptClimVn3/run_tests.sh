#!/usr/bin/env bash
# run all tests. First run . setup to set up search paths.
# Check that OPTCLIMTOP is set. If not, exit and ask user to set it.
if [[ -z "$OPTCLIMTOP" ]]; then
    echo "OPTCLIMTOP is not set. Please run the appropriate setup file."
    exit 1
fi

# then check if PROJECT_CODE setup
if [[ -z "$PROJECT_CODE" ]]; then
     echo "PROJECT_CODE is not set. Please set it to the project you are running under or to a dummy value. This is needed for some tests to run."
    exit 1
fi
export PYTHONPATH=$PYTHONPATH:$OPTCLIMTOP/OptClimVn3/test_OptClimVN3:$OPTCLIMTOP/OptClimVn3/Models/test_Models
for f in ${OPTCLIMTOP}/OptClimVn3/Models/test_Models/test_*.py  ${OPTCLIMTOP}/OptClimVn3/test_OptClimVN3/test_*.py \
		      ${OPTCLIMTOP}/OptClimVn3/support/test_support/test_*.py
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
