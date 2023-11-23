/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CumSumLayer.h"

#include "OperationUtils.h"

#include <cker/operation/optimized/CumSum.h>

namespace
{
using namespace onert;
using namespace onert::backend;
using namespace onert::backend::cpu::ops;

// This function assumes that layout of model is NHWC
int32_t getAxisValue(const IPortableTensor *axis)
{
  switch (axis->data_type())
  {
    case ir::DataType::INT32:
    {
      assert(axis->total_size() == sizeof(int32_t));
      return *getBuffer<int32_t>(axis);
    }
    case ir::DataType::INT64:
    {
      assert(axis->total_size() == sizeof(int64_t));
      return static_cast<int32_t>(*getBuffer<int64_t>(axis));
    }
    default:
      throw std::runtime_error("getAxis: Not supported data type");
  }
}

} // namespace

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

CumSumLayer::CumSumLayer()
  : _input(nullptr), _axis(nullptr), _output(nullptr), _exclusive(false), _reverse(false)
{
  // DO NOTHING
}

void CumSumLayer::configure(const IPortableTensor *input, const IPortableTensor *axis,
                            bool exclusive, bool reverse, IPortableTensor *output)
{
  _input = input;
  _axis = axis;
  _exclusive = exclusive;
  _reverse = reverse;
  _output = output;
}

void CumSumLayer::run()
{
  int axis = getAxisValue(_axis);
  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      nnfw::cker::optimized::CumSum<float>(getBuffer<float>(_input), getShape(_input), axis,
                                           _exclusive, _reverse, getBuffer<float>(_output));
      break;
    case OperandType::INT32:
      nnfw::cker::optimized::CumSum<int32_t>(getBuffer<int32_t>(_input), getShape(_input), axis,
                                             _exclusive, _reverse, getBuffer<int32_t>(_output));
      break;
    case OperandType::INT64:
      nnfw::cker::optimized::CumSum<int64_t>(getBuffer<int64_t>(_input), getShape(_input), axis,
                                             _exclusive, _reverse, getBuffer<int64_t>(_output));
      break;
    default:
      throw std::runtime_error{"CumSumLayer: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
