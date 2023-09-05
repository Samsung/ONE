/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ErrorMetric.h"

#include <loco/IR/DataType.h>
#include <loco/IR/DataTypeTraits.h>

#include <cmath>
#include <cassert>

using namespace mpqsolver::core;

/**
 * @brief compare first and second operands in MAE (Mean Average Error metric)
 */
float MAEMetric::compute(const WholeOutput &first, const WholeOutput &second) const
{
  assert(first.size() == second.size());

  float error = 0.f;
  size_t output_size = 0;

  for (size_t sample_index = 0; sample_index < first.size(); ++sample_index)
  {
    assert(first[sample_index].size() == second[sample_index].size());
    for (size_t out_index = 0; out_index < first[sample_index].size(); ++out_index)
    {
      const Buffer &first_elementary = first[sample_index][out_index];
      const Buffer &second_elementary = second[sample_index][out_index];
      assert(first_elementary.size() == second_elementary.size());
      size_t cur_size = first_elementary.size() / loco::size(loco::DataType::FLOAT32);

      const float *first_floats = reinterpret_cast<const float *>(first_elementary.data());
      const float *second_floats = reinterpret_cast<const float *>(second_elementary.data());
      for (size_t index = 0; index < cur_size; index++)
      {
        float ref_value = *(first_floats + index);
        float cur_value = *(second_floats + index);
        error += std::fabs(ref_value - cur_value);
      }
      output_size += cur_size;
    }
  }

  if (output_size == 0)
  {
    throw std::runtime_error("nothing to compare");
  }

  return error / output_size;
}
