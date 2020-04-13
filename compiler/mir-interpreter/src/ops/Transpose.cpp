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

#include "Transpose.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

#include "Common.h"

namespace mir_interpreter
{

template <typename T> struct TransposeImpl
{
  static void run(const mir::TensorVariant &input, const mir::ops::TransposeOp &op,
                  mir::TensorVariant &output);
};

template <typename T>
void TransposeImpl<T>::run(const mir::TensorVariant &inputv, const mir::ops::TransposeOp &op,
                           mir::TensorVariant &outputv)
{
  const auto &output_shape = op.getOutputShape(0);
  const auto &axis_order = op.getAxisOrder();
  const int32_t num_axis = static_cast<int32_t>(axis_order.size());
  assert(num_axis == inputv.getShape().rank());
  assert(num_axis == output_shape.rank());

  mir::Index output_index;
  output_index.resize(num_axis);

  mir::Tensor<T> input(inputv);
  mir::Tensor<T> output(outputv);

  for (auto &input_index : mir::ShapeRange(input.getShape()))
  {
    for (int32_t i = 0; i < num_axis; i++)
      output_index.at(i) = input_index.at(axis_order[i]);

    output.at(output_index) = input.at(input_index);
  }
}

void Transpose(const mir::TensorVariant &input, const mir::ops::TransposeOp &op,
               mir::TensorVariant &output)
{
  dispatch<TransposeImpl>(input.getElementType(), input, op, output);
}

} // namespace mir_interpreter
