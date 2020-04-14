#!/bin/bash

while IFS=',' read -r VERSION URL
do
  echo "Download ${VERSION} from '${URL}'"
  mkdir -p "${VERSION}"
  wget -nv -O "${VERSION}/schema.fbs" "${URL}"
  echo "Download ${VERSION} from '${URL}' - Done"
done < <(cat SCHEMA.lst | tail -n +2)
