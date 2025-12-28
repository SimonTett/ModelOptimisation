#!/bin/bash --login
export CYLC_VERSION=8
cylc play --no-run-name  -v -v  ~/optclim_runs/opt_cases/opt_dfols4/d400k/workflow --workflow-name=opt_dfols4/d400k
