/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Pad.h"
#include "Common.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

namespace mir_interpreter
{

using namespace mir;

template <typename T> struct PadImpl
{
  static void run(const mir::TensorVariant &inputv, const mir::ops::PadOp &op,
                  mir::TensorVariant &result);
};

template <typename T>
void PadImpl<T>::run(const TensorVariant &inputv, const ops::PadOp &op, TensorVariant &result)
{
  Tensor<T> result_accessor(result);
  Tensor<T> input(inputv);

  Shape out_shape = result_accessor.getShape();

  ShapeRange out_range(out_shape);
  const int rank = op.getInputShape(0).rank();
  const auto &padding_before = op.getPaddingBefore();
  const auto &padding_after = op.getPaddingAfter();

  Index temp_index;
  temp_index.resize(rank);

  bool index_on_padding(false);
  for (const Index &ind : out_range)
  {
    index_on_padding = false;

    for (int32_t i = 0; i < rank; i++)
    {
      // index on input values
      if (ind.at(i) >= padding_before[i] && ind.at(i) < out_shape.dim(i) - padding_after[i])
      {
        temp_index.at(i) = ind.at(i) - padding_before[i];
      }
      else
      { // not in input
        index_on_padding = true;
        break;
      }
    }
    if (index_on_padding)
    {
      result_accessor.at(ind) = op.getPaddingValue();
    }
    else
    {
      result_accessor.at(ind) = input.at(temp_index);
    }
  }
}

void Pad(const mir::TensorVariant &input, const mir::ops::PadOp &op, mir::TensorVariant &result)
{
  dispatch<PadImpl>(input.getElementType(), input, op, result);
};

} // namespace mir_interpreter
