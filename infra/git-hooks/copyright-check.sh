#!/usr/bin/env bash

CORRECT_COPYRIGHT="Copyright \(c\) [0-9\-]+ Samsung Electronics Co\., Ltd\. All Rights Reserved"

if [ $# -eq 0 ]; then
  echo "Please pass file(s) to check copyright"
  exit 1
fi

for f in "$@"; do
  if ! grep -qE "$CORRECT_COPYRIGHT" "$f"; then
    CREATED_YEAR=$(git log --follow --format=%aD "$f" | tail -1 | awk '{print $4}')
    EXAMPLE_COPYRIGHT="Copyright (c) $CREATED_YEAR Samsung Electronics Co., Ltd. All Rights Reserved"
    echo "Copyright format of $f is incorrect: recommend \"$EXAMPLE_COPYRIGHT\""
    INVALID_EXIT=1
  fi
done

if [[ $INVALID_EXIT -ne 0 ]]; then
    exit 1
fi
