/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _NNC_CORE_BACKEND_INTERPRETER_REDUCE_MEAN_
#define _NNC_CORE_BACKEND_INTERPRETER_REDUCE_MEAN_

#include "ReduceMean.h"
#include "Common.h"

#include "mir/ops/ReduceMeanOp.h"
#include "mir/Tensor.h"
#include "mir/ShapeRange.h"

namespace mir_interpreter
{

template <typename T> struct ReduceMeanImpl
{
  static void run(const mir::TensorVariant &inputv, const mir::ops::ReduceMeanOp &op,
                  mir::TensorVariant &output);
};

template <typename T>
void ReduceMeanImpl<T>::run(const mir::TensorVariant &inputv, const mir::ops::ReduceMeanOp &op,
                            mir::TensorVariant &output)
{
  const auto &input_shape = op.getInputShape(0);
  const auto &output_shape = op.getOutputShape(0);
  const auto &reduction_dims = op.getReductionDims();
  const bool keep_dims = op.getKeepDims();

  const auto reductor = [](T result, T x) { return result + x; };

  mir::Tensor<T> input(inputv);
  mir::Tensor<T> res_accessor(output);

  erase<T>(output);

  // This mask contains 'true' for dimensions that should be reduced. For example, if we want
  // to reduce dimensions 1 and 3 with total number of dimensions of 4, the mask will be
  // [false, true, false, true].
  std::vector<bool> reduction_dims_mask(input_shape.rank(), false);
  for (const int dim : reduction_dims)
  {
    reduction_dims_mask[dim] = true;
  }

  mir::Index out_index(output_shape.rank());
  for (const mir::Index &in_index : mir::ShapeRange(input_shape))
  {
    int out_index_dim = 0;
    for (int dim = 0; dim < input_shape.rank(); ++dim)
    {
      if (keep_dims)
      {
        out_index.at(out_index_dim++) = reduction_dims_mask[dim] ? 0 : in_index.at(dim);
      }
      else
      {
        if (!reduction_dims_mask[dim])
        {
          out_index.at(out_index_dim++) = in_index.at(dim);
        }
      }
    }
    res_accessor.at(out_index) = reductor(res_accessor.at(out_index), input.at(in_index));
  }

  const std::int32_t reduction_factor = input_shape.numElements() / output_shape.numElements();

  for (const auto &index : mir::ShapeRange(output_shape))
  {
    res_accessor.at(index) /= reduction_factor;
  }
}

void ReduceMean(const mir::TensorVariant &input, const mir::ops::ReduceMeanOp &op,
                mir::TensorVariant &output)
{
  dispatch<ReduceMeanImpl>(input.getElementType(), input, op, output);
};

} // namespace mir_interpreter

#endif //_NNC_CORE_BACKEND_INTERPRETER_REDUCE_MEAN_
