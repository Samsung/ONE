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

#include "nnkit/support/tflite/TensorUtils.h"

namespace nnkit
{
namespace support
{
namespace tflite
{

nncc::core::ADT::tensor::Shape tensor_shape(const TfLiteTensor *t)
{
  nncc::core::ADT::tensor::Shape shape;

  const int rank = t->dims->size;

  shape.resize(rank);
  for (int axis = 0; axis < rank; ++axis)
  {
    shape.dim(axis) = t->dims->data[axis];
  }

  return shape;
}

} // namespace tflite
} // namespace support
} // namespace nnkit
