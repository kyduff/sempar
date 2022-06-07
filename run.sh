#!/bin/bash

# check an image has been specified
[[ $# -ne 1 ]] && echo "usage: $(basename $0) <image>" && exit 1

# check whether to use GPU
if ! command -v nvidia-docker &> /dev/null
then
  cmd=docker
else
  cmd=nvidia-docker
fi

# run image in container
$cmd run -it \
  -v /users/ms21kjd/projects/sempar:/workspace/sempar \
  ${@:1}