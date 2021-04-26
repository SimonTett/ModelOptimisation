#!/usr/bin/env bash
#$ -N DumModel
#$ -l h_vmem=2G
#$ -l h_rt=00:10:00
# dummy script to simulate model running. We will just do ls and echo
# and run python code to generate fake model.
# then release the next script -- which is post processing script.
# all the post processing script will do is copy a file!
echo "Hello from dummy model"
echo "My path is "$(pwd)
mkdir W
mkdir A
fakemodel.py # run the fake model.
if [[ $? -ne 0 ]] ; then # failed
    echo "Failed to run model"
    echo "Exiting"
    exit 1
fi
sleep 5 # pretend to be a real model.
release=$(cat jobid.txt)
echo "Releasing $release"
ssh login01.ecdf.ed.ac.uk qrls $release
