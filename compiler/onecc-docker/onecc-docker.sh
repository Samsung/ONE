#!/usr/bin/env bash

if ! command -v docker > /dev/null
        then
        echo "docker must be installed"
        exit 0
fi

if ! command -v python3 > /dev/null
        then
        echo "python3 must be installed"
        exit 0
fi

if command -v onecc > /dev/null 
        then
        version=( $(onecc --version) )
        exec "python3" "onecc-docker.py" "${version[2]}" "$@" 
        exit 0
else
        exec "python3" "onecc-docker.py" "0" "$@" 
        exit 0
fi

