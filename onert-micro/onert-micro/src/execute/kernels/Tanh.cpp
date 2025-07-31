/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "execute/kernels/MathCommon.h"
#include "PALTanh.h"

using namespace onert_micro;
using namespace onert_micro::execute;

// NOTE: doesnt currently support dynamic shapes
namespace onert_micro
{
namespace execute
{

OMStatus execute_kernel_CircleTanh(const OMExecuteArgs &execute_args)
{
  const circle::Tensor *input = nullptr;
  const circle::Tensor *output = nullptr;

  uint8_t *input_data = nullptr;
  uint8_t *output_data = nullptr;

  SISOHeader(execute_args, &input, &output, &input_data, &output_data);

  OMStatus status;
  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
    {
      status =
        pal::Tanh<float>(core::OMRuntimeShape(input), reinterpret_cast<const float *>(input_data),
                         core::OMRuntimeShape(output), reinterpret_cast<float *>(output_data));
    }
    break;
#endif // DIS_FLOAT
#ifndef DIS_QUANT
    case circle::TensorType_INT8:
    {
      // TODO support CWQ
      onert_micro::core::QuantizationParams in_qparams = {
        (*input->quantization()->scale())[0],
        static_cast<int32_t>((*input->quantization()->zero_point())[0])};
      onert_micro::core::QuantizationParams out_qparams = {
        (*output->quantization()->scale())[0],
        static_cast<int32_t>((*output->quantization()->zero_point())[0])};

      status = pal::QuantizedTanh<int8_t>(
        core::OMRuntimeShape(input), in_qparams, reinterpret_cast<const int8_t *>(input_data),
        core::OMRuntimeShape(output), out_qparams, reinterpret_cast<int8_t *>(output_data));
    }
#endif // DIS_QUANT
  }

  return status;
}

} // namespace execute
} // namespace onert_micro
