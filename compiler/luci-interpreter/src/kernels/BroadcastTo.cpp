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

#include "kernels/BroadcastTo.h"
#include "kernels/Utils.h"

#include "PALBroadcastTo.h"

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

namespace
{

// TODO Extract this function to Utils.h
Shape extractShapeFromTensor(const Tensor *tensor)
{
  Shape shape(tensor->shape().num_elements());

  // Ensures the shape is 1D tensor
  LUCI_INTERPRETER_CHECK(tensor->shape().num_dims() == 1);

  if (tensor->element_type() == DataType::S32)
  {
    const auto *shape_data = tensor->data<int32_t>();
    for (int i = 0; i < tensor->shape().num_elements(); ++i)
    {
      // Ensures the dim value of shape is positive.
      LUCI_INTERPRETER_CHECK(shape_data[i] >= 0);

      shape.dim(i) = shape_data[i];
    }
  }
  else if (tensor->element_type() == DataType::S64)
  {
    const auto *shape_data = tensor->data<int64_t>();
    for (int i = 0; i < tensor->shape().num_elements(); ++i)
    {
      // Ensures the dim value of shape is positive.
      LUCI_INTERPRETER_CHECK(shape_data[i] >= 0);

      shape.dim(i) = static_cast<int32_t>(shape_data[i]);
      // Check value overflow
      LUCI_INTERPRETER_CHECK(static_cast<int64_t>(shape.dim(i)) == shape_data[i]);
    }
  }
  else
  {
    LUCI_INTERPRETER_CHECK(false);
  }
  return shape;
}

} // namespace

BroadcastTo::BroadcastTo(const Tensor *input, const Tensor *shape, Tensor *output)
  : Kernel({input, shape}, {output})
{
}

void BroadcastTo::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());

  Shape output_shape = extractShapeFromTensor(shape());

  int input_rank = input()->shape().num_dims();
  int output_rank = output_shape.num_dims();

  // Ensures output rank is not less than input rank
  LUCI_INTERPRETER_CHECK(input_rank <= output_rank);

  // Check if output shape is broadcastable from input shape
  // from https://www.tensorflow.org/api_docs/python/tf/broadcast_to
  // if a tensor has fewer axes than necessary its shape is padded on the left with ones.
  int extending_rank = output_rank - input_rank;
  for (int idx = 0; idx < input_rank; ++idx)
  {
    LUCI_INTERPRETER_CHECK(input()->shape().dim(idx) == 1 ||
                           input()->shape().dim(idx) == output_shape.dim(extending_rank + idx));
  }

  output()->resize(output_shape);
}

void BroadcastTo::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    default:
      throw std::runtime_error("luci-intp BroadcastTo Unsupported type.");
  }
}

void BroadcastTo::evalFloat() const
{
  luci_interpreter_pal::BroadcastTo(getTensorShape(input()), getTensorData<char>(input()),
                                    getTensorShape(output()), getTensorData<char>(output()),
                                    TfLiteType::kTfLiteFloat32);
}

} // namespace kernels
} // namespace luci_interpreter
