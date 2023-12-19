/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "metric.h"
#include "nnfw_util.h"

#include <algorithm>
#include <stdexcept>

namespace onert_train
{

Metric::Metric(const std::vector<Allocation> &output, const std::vector<Allocation> &expected,
               const std::vector<nnfw_tensorinfo> &infos)
  : _output{output}, _expected{expected}, _infos{infos}
{
  // DO NOTHING
}

template <typename T>
float Metric::categoricalAccuracy(const T *output, const T *expected, uint32_t batch, uint64_t size)
{
  int correct = 0;
  for (int b = 0; b < batch; ++b)
  {
    int begin_offset = b * size;
    int end_offset = begin_offset + size;
    std::vector<T> boutput(output + begin_offset, output + end_offset);
    std::vector<T> bexpected(expected + begin_offset, expected + end_offset);
    auto output_idx =
      std::distance(boutput.begin(), std::max_element(boutput.begin(), boutput.end()));
    auto expected_idx =
      std::distance(bexpected.begin(), std::max_element(bexpected.begin(), bexpected.end()));
    if (output_idx == expected_idx)
      correct++;
  }
  return static_cast<float>(correct) / batch;
}

float Metric::categoricalAccuracy(int32_t index)
{
  auto batch = _infos[index].dims[0];
  auto size = num_elems(&_infos[index]) / batch;
  switch (_infos[index].dtype)
  {
    case NNFW_TYPE_TENSOR_FLOAT32:
      return categoricalAccuracy(static_cast<const float *>(_output[index].data()),
                                 static_cast<const float *>(_expected[index].data()), batch, size);
    default:
      throw std::runtime_error("Not supported tensor type in calculateAccuracy");
  }
}

} // namespace onert_train
