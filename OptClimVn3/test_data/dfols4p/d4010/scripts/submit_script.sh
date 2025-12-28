#!/bin/bash --login
export CYLC_VERSION=8
cylc vip --no-run-name  -v -v  ~/optclim_runs/opt_cases/opt_dfols4/d4010/workflow --workflow-name=opt_dfols4/d4010
