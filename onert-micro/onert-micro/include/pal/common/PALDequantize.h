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

#ifndef ONERT_MICRO_EXECUTE_PAL_DEQUANTIZE_COMMON_H
#define ONERT_MICRO_EXECUTE_PAL_DEQUANTIZE_COMMON_H

#include "core/OMRuntimeShape.h"
#include "OMStatus.h"
#include "core/OMKernelData.h"
#include "PALUtils.h"

#include <cmath>

namespace onert_micro
{
namespace execute
{
namespace pal
{

template <typename InputT, typename OutputT>
OMStatus Dequantize(const core::QuantizationParams op_params, const uint32_t flat_size,
                    const InputT *input_data, OutputT *output_data)
{
  const int32_t zero_point = op_params.zero_point;
  const double scale = op_params.scale;

  for (uint32_t i = 0; i < flat_size; i++)
  {
    const int32_t val = input_data[i];
    const auto result = static_cast<OutputT>(scale * (val - zero_point));
    output_data[i] = result;
  }
  return Ok;
}
} // namespace pal
} // namespace execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_PAL_DEQUANTIZE_COMMON_H
