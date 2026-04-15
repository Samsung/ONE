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

#include "Convert.h"

#include <nncc/core/ADT/tensor/Shape.h>

#include <schema_generated.h>

using namespace nncc::core::ADT;

namespace tflimport
{

IndexVector as_index_vector(const flatbuffers::Vector<int32_t> *array)
{
  const uint32_t size = array->size();

  std::vector<int32_t> res(size);

  for (uint32_t i = 0; i < size; i++)
  {
    res[i] = array->Get(i);
  }

  return res;
}

tensor::Shape as_tensor_shape(const flatbuffers::Vector<int32_t> *shape)
{
  const uint32_t rank = shape->size();

  tensor::Shape res;

  res.resize(rank);
  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    res.dim(axis) = shape->Get(axis);
  }

  return res;
}

} // namespace tflimport
