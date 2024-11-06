#!/usr/bin/env bash
# Script to submit UM rose config  to remote machine
# Not properly implemented so will raise an error.
# takes model_dir & remote_machine as an argument
model_dir=$1 ; shift
if [[ $# > 0 ]]; then
  remote_machine=$1 ; shift
else
  remote_machine=''

echo "Submitting ${model_dir} to ${remote_machine}"
echo "Script not implemented. Exiting"
exit 1

# check have ssh stuff set up.
# rsync configuration to puma
# ssh run the rose command to actually submit it!