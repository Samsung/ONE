#!/bin/bash
set -v -e -x

if [ $# -ne 1 ] 
then
  echo "expected path to model"
  exit 1
fi

model_path="$1"
base_name="$(basename ${model_path} .tflite)"
proj_root="$(pwd)"

${proj_root}/build/compiler/tflite2circle/tflite2circle "${model_path}" "${base_name}.circle"
${proj_root}/build/compiler/circle2circle/circle2circle "${base_name}.circle" "${base_name}.opt.circle" --fold_cast --resolve_customop_max_pool_with_argmax
${proj_root}/compiler/record-minmax-conversion-test/gen_h5_random_inputs.py --model "${model_path}" --num_data=100 --output=random_dataset.h5
${proj_root}/build/compiler/circle-quantizer/circle-quantizer --quantize_dequantize_weights float32 int16 channel "${base_name}.opt.circle" "${base_name}.fake.q16.circle"
${proj_root}/build/compiler/record-minmax/record-minmax --input_model "${base_name}.fake.q16.circle" --output_model "${base_name}.populated.fake.q16.circle" --input_data random_dataset.h5 --min_percentile 0.0 --max_percentile 100.0
${proj_root}/build/compiler/circle-quantizer/circle-quantizer --quantize_with_minmax float32 int16 channel "${base_name}.populated.fake.q16.circle" "${base_name}.q16.circle"

