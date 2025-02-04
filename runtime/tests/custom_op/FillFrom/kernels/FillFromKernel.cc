/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnfw_experimental.h"

#include "flatbuffers/flexbuffers.h"

#include <stdexcept>
#include <limits>

extern "C" void FillFromEval(nnfw_custom_kernel_params *params, char *userdata,
                             size_t userdata_size)
{
  auto userdata_root = flexbuffers::GetRoot(reinterpret_cast<uint8_t *>(userdata), userdata_size);

  auto attr_map = userdata_root.AsMap();

  auto idx = attr_map["idx"].AsInt32();
  auto val = attr_map["val"].AsFloat();

  int32_t flat_size = 1;
  for (int32_t i = 0; i < params->inputs[0].type.rank; ++i)
  {
    flat_size *= params->inputs[0].type.dims[i];
  }

  if (!(0 <= idx && idx < flat_size))
    throw std::runtime_error("Value of attr Idx is out of range");

  auto output_flat = static_cast<float *>(params->outputs[0].allocation);
  auto input_flat = static_cast<float *>(params->inputs[0].allocation);

  for (int32_t i = 0; i < idx; ++i)
  {
    output_flat[i] = input_flat[i];
  }

  for (int32_t i = idx; i < flat_size; ++i)
  {
    output_flat[i] = val;
  }
}
