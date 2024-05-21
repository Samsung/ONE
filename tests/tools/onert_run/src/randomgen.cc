/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "randomgen.h"
#include "nnfw.h"
#include "nnfw_util.h"
#include "benchmark/RandomGenerator.h"

#include <iostream>

namespace onert_run
{

template <class T> void randomData(benchmark::RandomGenerator &randgen, void *data, uint64_t size)
{
  for (uint64_t i = 0; i < size; i++)
    reinterpret_cast<T *>(data)[i] = randgen.generate<T>();
}

void RandomGenerator::generate(std::vector<Allocation> &inputs)
{
  // generate random data
  const int seed = 1;
  benchmark::RandomGenerator randgen{seed, 0.0f, 2.0f};
  for (uint32_t i = 0; i < inputs.size(); ++i)
  {
    nnfw_tensorinfo ti;
    NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session_, i, &ti));
    auto input_size_in_bytes = bufsize_for(&ti);
    inputs[i].alloc(input_size_in_bytes);
    switch (ti.dtype)
    {
      case NNFW_TYPE_TENSOR_FLOAT32:
        randomData<float>(randgen, inputs[i].data(), num_elems(&ti));
        break;
      case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
        randomData<uint8_t>(randgen, inputs[i].data(), num_elems(&ti));
        break;
      case NNFW_TYPE_TENSOR_BOOL:
        randomData<bool>(randgen, inputs[i].data(), num_elems(&ti));
        break;
      case NNFW_TYPE_TENSOR_UINT8:
        randomData<uint8_t>(randgen, inputs[i].data(), num_elems(&ti));
        break;
      case NNFW_TYPE_TENSOR_INT32:
        randomData<int32_t>(randgen, inputs[i].data(), num_elems(&ti));
        break;
      case NNFW_TYPE_TENSOR_INT64:
        randomData<int64_t>(randgen, inputs[i].data(), num_elems(&ti));
        break;
      case NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED:
        randomData<int16_t>(randgen, inputs[i].data(), num_elems(&ti));
        break;
      default:
        std::cerr << "Not supported input type" << std::endl;
        std::exit(-1);
    }
    NNPR_ENSURE_STATUS(
      nnfw_set_input(session_, i, ti.dtype, inputs[i].data(), input_size_in_bytes));
    NNPR_ENSURE_STATUS(nnfw_set_input_layout(session_, i, NNFW_LAYOUT_CHANNELS_LAST));
  }
};

} // end of namespace onert_run
