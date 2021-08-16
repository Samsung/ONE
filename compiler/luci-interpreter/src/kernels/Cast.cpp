/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Cast.h"
#include "kernels/Utils.h"

namespace
{

using namespace luci_interpreter;
using namespace luci_interpreter::kernels;

template <typename InT, typename OutT>
void cast_data(const InT *in_data, OutT *out_data, uint32_t elements_count)
{
  std::transform(in_data, in_data + elements_count, out_data,
                 [](InT a) { return static_cast<OutT>(a); });
}

template <typename InT> void cast_from_pointer_to_tensor(const InT *in_data, Tensor *out_tensor)
{
  auto const out_type = out_tensor->element_type();
  auto const elements_count = out_tensor->shape().num_elements();

  switch (out_type)
  {
    case loco::DataType::U8:
      cast_data(in_data, getTensorData<uint8_t>(out_tensor), elements_count);
      break;
    case loco::DataType::U16:
      cast_data(in_data, getTensorData<uint16_t>(out_tensor), elements_count);
      break;
    case loco::DataType::U32:
      cast_data(in_data, getTensorData<uint32_t>(out_tensor), elements_count);
      break;
    case loco::DataType::U64:
      cast_data(in_data, getTensorData<uint64_t>(out_tensor), elements_count);
      break;
    case loco::DataType::S8:
      cast_data(in_data, getTensorData<int8_t>(out_tensor), elements_count);
      break;
    case loco::DataType::S16:
      cast_data(in_data, getTensorData<int16_t>(out_tensor), elements_count);
      break;
    case loco::DataType::S32:
      cast_data(in_data, getTensorData<int32_t>(out_tensor), elements_count);
      break;
    case loco::DataType::S64:
      cast_data(in_data, getTensorData<int64_t>(out_tensor), elements_count);
      break;
    case loco::DataType::FLOAT32:
      cast_data(in_data, getTensorData<float>(out_tensor), elements_count);
      break;
    case loco::DataType::BOOL:
      cast_data(in_data, getTensorData<bool>(out_tensor), elements_count);
      break;
    default:
      throw std::runtime_error("Unsupported output type.");
  }
}

void cast_from_tensor_to_tensor(const Tensor *in_tensor, Tensor *out_tensor)
{
  auto in_type = in_tensor->element_type();

  switch (in_type)
  {
    case loco::DataType::U8:
      cast_from_pointer_to_tensor(getTensorData<uint8_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::U16:
      cast_from_pointer_to_tensor(getTensorData<uint16_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::U32:
      cast_from_pointer_to_tensor(getTensorData<uint32_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::U64:
      cast_from_pointer_to_tensor(getTensorData<uint64_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::S8:
      cast_from_pointer_to_tensor(getTensorData<int8_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::S16:
      cast_from_pointer_to_tensor(getTensorData<int16_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::S32:
      cast_from_pointer_to_tensor(getTensorData<int32_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::S64:
      cast_from_pointer_to_tensor(getTensorData<int64_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::FLOAT32:
      cast_from_pointer_to_tensor(getTensorData<float>(in_tensor), out_tensor);
      break;
    case loco::DataType::BOOL:
      cast_from_pointer_to_tensor(getTensorData<bool>(in_tensor), out_tensor);
      break;
    default:
      throw std::runtime_error("Unsupported input type.");
  }
}

bool is_quantized(const Tensor *t)
{
  return t->scales().size() != 0;
}

template <typename InT, typename OutT>
void cast_to_quant_data(const InT *in_data, OutT *out_data, uint32_t elements_count, float scale, int32_t zerop)
{
  std::transform(in_data, in_data + elements_count, out_data,
                 [scale, zerop](InT a) { return static_cast<OutT>(static_cast<OutT>(static_cast<float>(a)/scale + zerop)); });
}

template <typename InT> void cast_to_quant_from_pointer_to_tensor(const InT *in_data, Tensor *out_tensor)
{
  auto const out_type = out_tensor->element_type();
  auto const elements_count = out_tensor->shape().num_elements();
  float scale = out_tensor->scale();
  int32_t zerop = out_tensor->zero_point();

  switch (out_type)
  {
    case loco::DataType::U8:
      cast_to_quant_data(in_data, getTensorData<uint8_t>(out_tensor), elements_count, scale, zerop);
      break;
    case loco::DataType::S8:
      cast_to_quant_data(in_data, getTensorData<int8_t>(out_tensor), elements_count, scale, zerop);
      break;
    case loco::DataType::U16:
      cast_to_quant_data(in_data, getTensorData<uint16_t>(out_tensor), elements_count, scale, zerop);
      break;
    case loco::DataType::S16:
      cast_to_quant_data(in_data, getTensorData<int16_t>(out_tensor), elements_count, scale, zerop);
      break;
    default:
      throw std::runtime_error("Unsupported output type");
  }
}

void cast_to_quant(const Tensor *in_tensor, Tensor *out_tensor)
{
  auto in_type = in_tensor->element_type();

  switch (in_type)
  {
    case loco::DataType::U8:
      cast_to_quant_from_pointer_to_tensor(getTensorData<uint8_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::U16:
      cast_to_quant_from_pointer_to_tensor(getTensorData<uint16_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::U32:
      cast_to_quant_from_pointer_to_tensor(getTensorData<uint32_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::U64:
      cast_to_quant_from_pointer_to_tensor(getTensorData<uint64_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::S8:
      cast_to_quant_from_pointer_to_tensor(getTensorData<int8_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::S16:
      cast_to_quant_from_pointer_to_tensor(getTensorData<int16_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::S32:
      cast_to_quant_from_pointer_to_tensor(getTensorData<int32_t>(in_tensor), out_tensor);
      break;
    case loco::DataType::S64:
      cast_to_quant_from_pointer_to_tensor(getTensorData<int64_t>(in_tensor), out_tensor);
      break;
    default:
      throw std::runtime_error("Unsupported input type");
  }
}

template <typename InT, typename OutT>
void cast_from_quant(const InT *in_data, OutT *out_data, uint32_t elements_count, float scale, int32_t zerop)
{
  std::transform(in_data, in_data + elements_count, out_data,
                 [scale, zerop](InT a) { return static_cast<OutT>((a-zerop)*scale); });
}

template <typename InT>
void cast_from_quant_from_pointer_to_tensor(const InT *in_data, Tensor *out_tensor, float scale, int32_t zerop)
{
  auto out_type = out_tensor->element_type();
  auto const elements_count = out_tensor->shape().num_elements();

  switch (out_type)
  {
    case loco::DataType::U8:
      cast_from_quant(in_data, getTensorData<uint8_t>(out_tensor), elements_count, scale, zerop);
      break;
    case loco::DataType::U16:
      cast_from_quant(in_data, getTensorData<uint16_t>(out_tensor), elements_count, scale, zerop);
      break;
    case loco::DataType::U32:
      cast_from_quant(in_data, getTensorData<uint32_t>(out_tensor), elements_count, scale, zerop);
      break;
    case loco::DataType::U64:
      cast_from_quant(in_data, getTensorData<uint64_t>(out_tensor), elements_count, scale, zerop);
      break;
    case loco::DataType::S8:
      cast_from_quant(in_data, getTensorData<int8_t>(out_tensor), elements_count, scale, zerop);
      break;
    case loco::DataType::S16:
      cast_from_quant(in_data, getTensorData<int16_t>(out_tensor), elements_count, scale, zerop);
      break;
    case loco::DataType::S32:
      cast_from_quant(in_data, getTensorData<int32_t>(out_tensor), elements_count, scale, zerop);
      break;
    case loco::DataType::S64:
      cast_from_quant(in_data, getTensorData<int64_t>(out_tensor), elements_count, scale, zerop);
      break;
    default:
      throw std::runtime_error("Unsupported output type");
  }
}

void cast_from_quant(const Tensor *in_tensor, Tensor *out_tensor)
{
  auto in_type = in_tensor->element_type();

  float scale = in_tensor->scale();
  int32_t zerop = in_tensor->zero_point();

  switch (in_type)
  {
    case loco::DataType::U8:
      cast_from_quant_from_pointer_to_tensor(getTensorData<uint8_t>(in_tensor), out_tensor, scale, zerop);
      break;
    case loco::DataType::S8:
      cast_from_quant_from_pointer_to_tensor(getTensorData<int8_t>(in_tensor), out_tensor, scale, zerop);
      break;
    case loco::DataType::U16:
      cast_from_quant_from_pointer_to_tensor(getTensorData<uint16_t>(in_tensor), out_tensor, scale, zerop);
      break;
    case loco::DataType::S16:
      cast_from_quant_from_pointer_to_tensor(getTensorData<int16_t>(in_tensor), out_tensor, scale, zerop);
      break;
    default:
      throw std::runtime_error("Unsupported input type");
  }
}

} // namespace

namespace luci_interpreter
{
namespace kernels
{

Cast::Cast(const Tensor *input, Tensor *output) : Kernel({input}, {output}) {}

void Cast::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() != loco::DataType::Unknown);
  LUCI_INTERPRETER_CHECK(output()->element_type() != loco::DataType::Unknown);

  const Shape &shape = input()->shape();
  output()->resize(shape);
}

void Cast::execute() const
{
  assert(input()->shape().num_elements() == output()->shape().num_elements());

  if (is_quantized(input()) && !is_quantized(output()))
  {
    cast_from_quant(input(), output());
  }
  else if (!is_quantized(input()) && is_quantized(output()))
  {
    cast_to_quant(input(), output());
  }
  else if (is_quantized(input()) && is_quantized(output()))
  {
    assert(false && "requantization is not implemented");
  }
  else
  {
    assert(!is_quantized(input()) && !is_quantized(output()));
    cast_from_tensor_to_tensor(input(), output());
  }
}

} // namespace kernels
} // namespace luci_interpreter
