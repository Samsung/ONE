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

#include "CastLayer.h"

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

CastLayer::CastLayer() : _input(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void CastLayer::configure(const IPortableTensor *input, IPortableTensor *output)
{
  _input = input;
  _output = output;
}

template <typename FromT, typename ToT> void CastLayer::castTensor(const FromT *in, ToT *out)
{
  auto input_shape = getTensorShape(_input);
  auto output_shape = getTensorShape(_output);
  const auto num_elements = MatchingFlatSize(input_shape, output_shape);

  std::transform(in, in + num_elements, out, [](FromT a) { return static_cast<ToT>(a); });
}

template <typename FromT> void CastLayer::castPtr(const FromT *in, DataPtr out)
{
  switch (_output->data_type())
  {
    case ir::DataType::FLOAT32:
      castTensor(in, out.f);
      return;
    case ir::DataType::INT32:
      castTensor(in, out.i32);
      return;
    case ir::DataType::UINT32:
      castTensor(in, out.u32);
      return;
    case ir::DataType::UINT8:
      castTensor(in, out.u8);
      return;
    case ir::DataType::BOOL8:
      castTensor(in, out.b);
      return;
    case ir::DataType::INT64:
      castTensor(in, out.i64);
      return;
    default:
      throw std::runtime_error("Not supported output type" +
                               std::to_string((int)_output->data_type()));
  }
}

void CastLayer::run()
{
  auto input_buf = _input->buffer();
  auto output_buf = _output->buffer();
  const auto in = *reinterpret_cast<const DataPtr *>(&input_buf);
  auto out = *reinterpret_cast<DataPtr *>(&output_buf);

  switch (_input->data_type())
  {
    case ir::DataType::FLOAT32:
      castPtr(in.f, out);
      return;
    case ir::DataType::INT32:
      castPtr(in.i32, out);
      return;
    case ir::DataType::UINT32:
      castPtr(in.u32, out);
      return;
    case ir::DataType::UINT8:
      castPtr(in.u8, out);
      return;
    case ir::DataType::BOOL8:
      castPtr(in.b, out);
      return;
    case ir::DataType::INT64:
      castPtr(in.i64, out);
      return;
    default:
      throw std::runtime_error("Cast: unsupported data type" +
                               std::to_string((int)_input->data_type()));
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
