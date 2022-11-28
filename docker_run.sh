#!/bin/sh
if [ $# -eq 0 ]
  then
    echo "No input arguments, launching bash."
    docker run --gpus all -v $(pwd):/agency -it agency bash
else
    echo "Running $@ in docker container using agency code from host machine."
    docker run --gpus all -v $(pwd):/agency -it agency "$@"
fi