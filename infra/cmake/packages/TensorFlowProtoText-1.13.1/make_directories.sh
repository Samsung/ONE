#!/bin/bash

while [[ $# -ne 0 ]]; do
  DIR=$1; shift
  mkdir -p "${DIR}"
done
