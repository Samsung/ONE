/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PowLayer.h"

#include <cker/operation/Pow.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

void PowLayer::configure(const operand::Tensor *lhs, const operand::Tensor *rhs,
                         operand::Tensor *output)
{
  _lhs = lhs;
  _rhs = rhs;
  _output = output;
}

void PowLayer::run()
{
  if (!HaveSameShapes(_lhs, _rhs))
  {
    throw std::runtime_error{"Pow NYI for broadcast."};
  }

  nnfw::cker::powImpl(
      convertTensorToCkerShape(_lhs), reinterpret_cast<const float *>(_lhs->buffer()),
      convertTensorToCkerShape(_rhs), reinterpret_cast<const float *>(_rhs->buffer()),
      convertTensorToCkerShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert
