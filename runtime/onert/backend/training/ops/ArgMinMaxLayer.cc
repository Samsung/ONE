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

#include "ArgMinMaxLayer.h"

#include "OperationUtils.h"

#include <cker/operation/ArgMinMax.h>
#include <assert.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{
namespace
{
template <typename T> std::function<bool(T, T)> GetComparefunction(bool is_arg_max)
{
  if (is_arg_max)
  {
    return std::greater<T>();
  }
  else
  {
    return std::less<T>();
  }
}
} // namespace

void ArgMinMaxLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                               const IPortableTensor *axis, bool is_arg_max)
{
  _input = input;
  _output = output;
  _axis = axis;
  _is_arg_max = is_arg_max;
}

void ArgMinMaxLayer::run()
{
  if (_axis->total_size() != sizeof(int32_t))
  {
    throw std::runtime_error("ArgMinMax: wrong shape of axis");
  }
  auto axis = *getBuffer<int32_t>(_axis);
  if (axis < 0)
  {
    axis += _input->getShape().rank();
  }
#define TF_LITE_ARG_MIN_MAX(input_type, axis_type, output_type)                 \
  ArgMinMax(getShape(_input), getBuffer<input_type>(_input), getShape(_output), \
            getBuffer<output_type>(_output), axis, GetComparefunction<input_type>(_is_arg_max));
  if (_output->data_type() == ir::DataType::INT32)
  {
    switch (_input->data_type())
    {
      case ir::DataType::FLOAT32:
        TF_LITE_ARG_MIN_MAX(float, int32_t, int32_t);
        break;
      case ir::DataType::QUANT_UINT8_ASYMM:
      case ir::DataType::UINT8:
        TF_LITE_ARG_MIN_MAX(uint8_t, int32_t, int32_t);
        break;
      case ir::DataType::QUANT_INT8_ASYMM:
        TF_LITE_ARG_MIN_MAX(uint8_t, int32_t, int32_t);
        break;
      case ir::DataType::INT32:
        TF_LITE_ARG_MIN_MAX(int32_t, int32_t, int32_t);
        break;
      default:
        throw std::runtime_error("ArgMinMax: unsupported data type");
    }
  }
  else if (_output->data_type() == ir::DataType::INT64)
  {
    switch (_input->data_type())
    {
      case ir::DataType::FLOAT32:
        TF_LITE_ARG_MIN_MAX(float, int32_t, int64_t);
        break;
      case ir::DataType::QUANT_UINT8_ASYMM:
      case ir::DataType::UINT8:
        TF_LITE_ARG_MIN_MAX(uint8_t, int32_t, int64_t);
        break;
      case ir::DataType::QUANT_INT8_ASYMM:
        TF_LITE_ARG_MIN_MAX(uint8_t, int32_t, int64_t);
        break;
      case ir::DataType::INT32:
        TF_LITE_ARG_MIN_MAX(int32_t, int32_t, int64_t);
        break;
      default:
        throw std::runtime_error("ArgMinMax: unsupported data type");
    }
  }
  else
  {
    throw std::runtime_error("ArgMinMax: unsupported data type");
  }

#undef TF_LITE_ARG_MIN_MAX
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
