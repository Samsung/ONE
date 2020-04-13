/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PermuteLayer.h"

namespace neurun
{
namespace backend
{
namespace cpu
{
namespace kernel
{

using Type = ir::operation::Permute::Type;

void PermuteLayer::configure(std::shared_ptr<backend::ITensor> input,
                             std::shared_ptr<backend::ITensor> output,
                             const ir::Shape &output_shape, Type type, ir::DataType dataType)
{
  _input = input;
  _output = output;
  _output_shape = output_shape;
  _type = type;
  _dataType = dataType;
}

void PermuteLayer::run()
{
  using ir::DataType;
  switch (_dataType)
  {
    case DataType::FLOAT32:
      runTempl<float>();
      break;
    case DataType::INT32:
      runTempl<int32_t>();
      break;
    case DataType::UINT32:
      runTempl<uint32_t>();
      break;
    case DataType::BOOL8:
    case DataType::QUANT8_ASYMM:
      runTempl<uint8_t>();
      break;
    case DataType::QUANT8_SYMM:
      runTempl<int8_t>();
      break;
    default:
      throw std::runtime_error("NYI");
      break;
  }
}

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace neurun
