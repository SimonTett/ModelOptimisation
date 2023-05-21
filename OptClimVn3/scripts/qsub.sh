#!/usr/bin/env bash
# script to be run by qsub to run next iteration.
# args are command and its arguments
# this only necessary as environment needs to be setup..
echo "I am qsub called with $*"
cmd=$1; shift
cmdargs=$*
echo "Current Dir  is $PWD"
echo "PythonPath is $PYTHONPATH"
echo "Path is $PATH"
echo "Command is $cmd with $cmdargs"
echo "========================"
stat=$($cmd $cmdargs)
echo "stat is $stat"
