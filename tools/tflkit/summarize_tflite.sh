#!/bin/bash

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
python "${SCRIPT_PATH}/../tflitefile_tool/model_parser.py" ${@}
