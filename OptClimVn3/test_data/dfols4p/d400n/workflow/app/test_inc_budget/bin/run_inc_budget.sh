#!/usr/bin/env bash

set -eu

# Load python environment including Iris
module load $IRISENV
module list 2>&1

# Run script
python test_inc_budget.py
