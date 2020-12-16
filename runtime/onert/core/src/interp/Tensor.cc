/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Tensor.h"

#define NO_USE(a) (void)(a)

namespace onert
{
namespace interp
{

void ITensor::access(const std::function<void(backend::ITensor &tensor)> &fn) { fn(*this); }

size_t ROTensor::calcOffset(const ir::Coordinates &coords) const
{
  NO_USE(coords);
  throw std::runtime_error("offset_element_in_bytes is not supported for cpu::Tensor now.");
}

size_t Tensor::calcOffset(const ir::Coordinates &coords) const
{
  NO_USE(coords);
  throw std::runtime_error("offset_element_in_bytes is not supported for cpu::Tensor now.");
}

ir::Layout ROTensor::layout() const
{
  // TODO Changes to return frontend layout
  return ir::Layout::NHWC;
}

ir::Layout Tensor::layout() const
{
  // TODO Changes to return frontend layout
  return ir::Layout::NHWC;
}

ir::Shape Tensor::getShape() const { return _info.shape(); }

ir::Shape ROTensor::getShape() const { return _info.shape(); }

} // namespace interp
} // namespace onert
