#!/bin/bash

set -eu

progname=$(basename "${BASH_SOURCE[0]}")
outdir="."
name=""
configs_src=()
models_src=()
models_conn_src=()
configs_str=""
models_str=""
types_str=""
pkg_inputs_str=""
pkg_outputs_str=""
model_conn_str=""

usage() {
  echo "Usage: $progname [options]"
  echo "Convert modelfile (tflite, circle or tvn) to nnpackage."
  echo ""
  echo "Options:"
  echo "    -h   show this help"
  echo "    -o   set nnpackage output directory (default=$outdir)"
  echo "    -p   set nnpackage output name (default=[1st modelfile name])"
  echo "    -c   provide configuration files"
  echo "    -m   provide model files"
  echo "    -i   provide files of models' connection information between models"
  echo "         This option is for multi-model. You don't need to use this option if you want to create nnpkg with a single model."
  echo ""
  echo "         (Will be deprecated: if there is one remain parameter, that is model file)"
  echo ""
  echo "Examples:"
  echo "    $progname -m add.tflite                                              => create nnpackage 'add' in $outdir/"
  echo "    $progname -o out -m add.tflite                                       => create nnpackage 'add' in out/"
  echo "    $progname -o out -p addpkg -m add.tflite                             => create nnpackage 'addpkg' in out/"
  echo "    $progname -c add.cfg -m add.tflite                                   => create nnpackage 'add' with add.cfg"
  echo "    $progname -o out -p addpkg -m a1.tflite a2.tflite -i a1.json a2.json => create nnpackage 'addpkg' with models a1.tflite and a2.tflite in out/"
  echo ""
  echo "(Will be deprecated: if there is one remain parameter, that is model file)"
  exit 1
}

if [ $# -eq 0 ]; then
  >&2 echo "For help, type $progname -h"
  exit 1
fi

while getopts "ho:p:c:m:i:" OPTION; do
  case "${OPTION}" in
    h) usage;;
    o) outdir=$OPTARG;;
    p) name=$OPTARG;;
    c)
      configs_src=($OPTARG)
      until [[ $OPTIND -gt $# ]] || [[ $(eval "echo \${$OPTIND}") =~ ^-.* ]] || [ -z $(eval "echo \${$OPTIND}") ]; do
        if [[ $OPTIND -eq $# ]] && [[ ${#models_src[@]} -eq 0 ]]; then
          # Backward compatibility (will be deprecated)
          # The last remain parameter is model if there is no option "-m"
          models_src=($(eval "echo \${$OPTIND}"))
        else
          configs_src+=($(eval "echo \${$OPTIND}"))
        fi
        OPTIND=$((OPTIND + 1))
      done
      ;;
    m)
      models_src=($OPTARG)
      until [[ $OPTIND -gt $# ]] || [[ $(eval "echo \${$OPTIND}") =~ ^-.* ]] || [ -z $(eval "echo \${$OPTIND}") ]; do
        models_src+=($(eval "echo \${$OPTIND}"))
        OPTIND=$((OPTIND + 1))
      done
      ;;
    i)
      models_conn_src=($OPTARG)
      until [[ $OPTIND -gt $# ]] || [[ $(eval "echo \${$OPTIND}") =~ ^-.* ]] || [ -z $(eval "echo \${$OPTIND}") ]; do
        models_conn_src+=("$(eval "echo \${$OPTIND}")")
        OPTIND=$((OPTIND + 1))
      done
      ;;
    ?) exit 1;;
  esac
done

shift $((OPTIND-1))

# Backward compatibility (will be deprecated)
# The last remain parameter is model if there is no option "-m"
if [ $# -eq 1 ] && [ ${#models_src[@]} -eq 0 ]; then
  models_src=($1)
  shift 1
fi

if [ $# -ne 0 ]; then
  >&2 echo "error: wrong argument (too many arguments)."
  >&2 echo "For help, type $progname -h"
  exit 1
fi

if [[ ${#configs_src[@]} -ne 0 ]] && [[ ${#configs_src[@]} -ne ${#models_src[@]} ]]; then
  >&2 echo "error: when config file is provided, # of config file should be same with modelfile"
  >&2 echo "Please provide config file for each model file, or don't provide config file."

fi

if [[ ${#models_conn_src[@]} -ne 0 ]] && [[ ${#models_conn_src[@]} -ne ${#models_src[@]} ]]; then
  >&2 echo "error: when model connection files are provided, # of model info files should be same with modelfile"
  exit 1
fi

delim=""
for modelpath in ${models_src[@]}
do
  modelfile=$(basename "$modelpath")

  if [[ "$modelfile" != *.* ]]; then
    >&2 echo "error: modelfile does not have extension."
    >&2 echo "Please provide extension so that $progname can identify what type of model you use."
    exit 1
  fi

  if [ ! -e $modelpath ]; then
    >&2 echo "error: "$modelpath" does not exist."
    exit 1
  fi

  models_str="$models_str$delim\"$modelfile\""
  types_str="$types_str$delim\"${modelfile##*.}\""
  delim=", "
done

org_models_io_json='{ "inputs" : [], "outputs" : [] }'
new_models_io_json='{ "inputs" : [], "outputs" : [] }'
for models_conn_path in "${models_conn_src[@]}"
do
  models_conn_file=$(basename "$models_conn_path")

  if [ ! -e $models_conn_file ]; then
    >&2 echo "error: "$models_conn_file" does not exist."
    exit 1
  fi

  # jq cannot compile keys with "-", so enclose those keys in double quotes
  inputs=$(jq -r '."org-model".inputs' $models_conn_path)
  org_models_io_json=$(echo $org_models_io_json | jq --argjson ins "$inputs" '.inputs += [$ins]')

  outputs=$(jq -r '."org-model".outputs' $models_conn_path)
  org_models_io_json=$(echo $org_models_io_json | jq --argjson outs "$outputs" '.outputs += [$outs]')

  inputs=$(jq -r '."new-model".inputs' $models_conn_path)
  new_models_io_json=$(echo $new_models_io_json | jq --argjson ins "$inputs" '.inputs += [$ins]')

  outputs=$(jq -r '."new-model".outputs' $models_conn_path)
  new_models_io_json=$(echo $new_models_io_json | jq --argjson outs "$outputs" '.outputs += [$outs]')
done

# Set string for model-conn
declare -A connect_map
in_model=0
for model_input in $(echo "${new_models_io_json}" | jq -c '.inputs[]'); do
  input_pos=0

  for org_input_index in $(echo "${model_input}" | jq '.org[]'); do
    out_model=0

    for model_output in $(echo "${new_models_io_json}" | jq -c '.outputs[]'); do
      if [[ $in_model == $out_model ]]; then
        continue
      fi

      output_pos=0

      for org_output_index in $(echo "${model_output}" | jq '.org[]'); do
        if [[ $org_input_index == $org_output_index ]]; then
          to=$(echo "$in_model:0:$input_pos" | jq -R)
          from=$(echo "$out_model:0:$output_pos" | jq -R)
          connect_map[$from]+="$to,"
        fi

        (( output_pos+=1 ))
      done

      (( out_model+=1 ))
    done

    (( input_pos+=1 ))
  done

  (( in_model+=1 ))
done

for from in ${!connect_map[@]}; do
  to=${connect_map[$from]%?}
  model_conn_str+="{ \"from\" : $from, \"to\" : [ $to ] },"
done
model_conn_str=${model_conn_str%?}

# Set string for pkg-inputs and pkg-outputs
function find_same_index_pos {
  for new_model_input in $(echo "$1" | jq -c '.[]'); do
    new_model_input_pos=0
    for new_model_org_input_index in $(echo "${new_model_input}" | jq '.org[]'); do
      if [[ org_input_index == new_model_org_input_index ]]; then
        return new_model_input_pos
      fi
      (( new_model_input_pos+=1 ))
    done
  done
}

pkg_inputs_model=()
pkg_inputs_io=()
model_index=0
for org_model_input in $(echo "${org_models_io_json}" | jq -c '.inputs[]'); do
  pkg_input_pos=0

  for new_input_index in $(echo "${org_model_input}" | jq '.new[]'); do
    if [[ $new_input_index != -1 ]]; then
      pkg_inputs_model[$pkg_input_pos]=$model_index

      new_model_inputs=$(echo "${new_models_io_json}" | jq -c '.inputs')
      org_model_input_index=$(echo "${org_model_input}" | jq --argjson pos "$pkg_input_pos" '.org[$pos]')
      find_same_index_pos "$new_model_inputs" "$org_model_input_index"
      new_model_input_pos=$?
      pkg_inputs_io[$pkg_input_pos]=$new_model_input_pos
    fi

    (( pkg_input_pos+=1 ))
  done

  (( model_index+=1 ))
done

for (( i = 0 ; i < ${#pkg_inputs_model[@]} ; i++ )) ; do
  pkg_inputs_str+="${pkg_inputs_model[$i]}:0:${pkg_inputs_io[$i]},"
done
pkg_inputs_str=${pkg_inputs_str%?}


pkg_outputs_model=()
pkg_outputs_io=()
model_index=0
for org_model_output in $(echo "${org_models_io_json}" | jq -c '.outputs[]'); do
  pkg_output_pos=0

  for new_output in $(echo "${org_model_output}" | jq '.new[]'); do
    if [[ $new_output != -1 ]]; then
      pkg_outputs_model[$pkg_output_pos]=$model_index

      new_model_outputs=$(echo "${new_models_io_json}" | jq -c '.outputs')
      org_model_output_index=$(echo "${org_model_output}" | jq --argjson pos "$pkg_output_pos" '.org[$pos]')
      find_same_index_pos "$new_model_outputs" "$org_model_output_index"
      new_model_output_pos=$?
      pkg_outputs_io[$pkg_output_pos]=$new_model_output_pos
    fi

    (( pkg_output_pos+=1 ))
  done

  (( model_index+=1 ))
done

for (( i = 0 ; i < ${#pkg_outputs_model[@]} ; i++ )) ; do
  pkg_outputs_str+="${pkg_outputs_model[$i]}:0:${pkg_outputs_io[$i]},"
done
pkg_outputs_str=${pkg_outputs_str%?}

delim=""
for configpath in ${configs_src[@]}
do
  configfile=$(basename "$configpath")

  if [ ! -e $configpath ]; then
    >&2 echo "error: "$configpath" does not exist."
    exit 1
  fi

  configs_str="$configs_str$delim\"$configfile\""
  delim=", "
done

if [ -z "$name" ]; then
  first_modelfile=$(basename "${models_src[0]}")
  name=${first_modelfile%.*}
fi

echo "$progname: Generating nnpackage "$name" in "$outdir""
mkdir -p "$outdir"/"$name"/metadata

if [[ $model_conn_str == "" ]]; then
  cat > "$outdir"/"$name"/metadata/MANIFEST <<-EOF
{
  "major-version" : "1",
  "minor-version" : "2",
  "patch-version" : "0",
  "configs"       : [ $configs_str ],
  "models"        : [ $models_str ],
  "model-types"   : [ $types_str ]
}
EOF
else
  cat > "$outdir"/"$name"/metadata/MANIFEST <<-EOF
{
  "major-version" : "1",
  "minor-version" : "2",
  "patch-version" : "0",
  "configs"       : [ $configs_str ],
  "models"        : [ $models_str ],
  "model-types"   : [ $types_str ],
  "pkg-inputs"    : [ $pkg_inputs_str ],
  "pkg-outputs"   : [ $pkg_outputs_str ],
  "model-connect" : [ $model_conn_str ]
}
EOF
fi

for modelpath in ${models_src[@]}
do
  cp "$modelpath" "$outdir"/"$name"
done

for configpath in ${configs_src[@]}
do
  cp "$configpath" "$outdir/$name/metadata"
done
