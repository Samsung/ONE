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
  if (first.size() != second.size())
  {
    throw std::runtime_error("Can not compare vectors of different sizes");
  }

  double output_errors = 0.; // mean over mean outputs errors
  size_t num_output_errors = 0;

  for (size_t sample_index = 0; sample_index < first.size(); ++sample_index)
  {
    assert(first[sample_index].size() == second[sample_index].size());
    for (size_t out_index = 0; out_index < first[sample_index].size(); ++out_index)
    {
      const Buffer &first_elementary = first[sample_index][out_index];
      const Buffer &second_elementary = second[sample_index][out_index];
      assert(first_elementary.size() == second_elementary.size());
      size_t cur_size = first_elementary.size() / loco::size(loco::DataType::FLOAT32);

      double output_error = 0.; // mean error oevr current output

      const float *first_floats = reinterpret_cast<const float *>(first_elementary.data());
      const float *second_floats = reinterpret_cast<const float *>(second_elementary.data());
      for (size_t index = 0; index < cur_size; index++)
      {
        double ref_value = static_cast<double>(*(first_floats + index));
        double cur_value = static_cast<double>(*(second_floats + index));
        output_error += std::fabs(ref_value - cur_value);
      }
      if (cur_size != 0)
      {
        output_errors += (output_error / cur_size);
        num_output_errors += 1;
      }
    }
  }

  if (num_output_errors == 0)
  {
    throw std::runtime_error("Nothing to compare");
  }

  return static_cast<float>(output_errors / num_output_errors);
}
