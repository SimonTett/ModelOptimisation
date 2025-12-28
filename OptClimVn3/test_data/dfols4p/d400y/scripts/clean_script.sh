#!/bin/bash --login
export CYLC_VERSION=8
cylc stop --now --now --max-polls=100 opt_dfols4/d400y
cylc clean --yes opt_dfols4/d400y
