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

#include "morph/nnapi.h"

#include <cassert>

using namespace nncc::core::ADT;

namespace morph
{
namespace nnapi
{

tensor::Shape as_tensor_shape(const feature::Shape &shape)
{
  tensor::Shape res;

  res.resize(4);
  res.dim(0) = 1;
  res.dim(1) = shape.height();
  res.dim(2) = shape.width();
  res.dim(3) = shape.depth();

  return res;
}

tensor::Shape as_tensor_shape(const kernel::Shape &shape)
{
  tensor::Shape res;

  res.resize(4);
  res.dim(0) = shape.count();
  res.dim(1) = shape.height();
  res.dim(2) = shape.width();
  res.dim(3) = shape.depth();

  return res;
}

feature::Shape as_feature_shape(const tensor::Shape &shape)
{
  assert(shape.rank() == 4);
  assert(shape.dim(0) == 1);
  return feature::Shape{shape.dim(3), shape.dim(1), shape.dim(2)};
}

kernel::Shape as_kernel_shape(const tensor::Shape &shape)
{
  assert(shape.rank() == 4);
  return kernel::Shape{shape.dim(0), shape.dim(3), shape.dim(1), shape.dim(2)};
}

} // namespace nnapi
} // namespace morph
