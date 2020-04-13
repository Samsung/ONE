/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Import/CircleReader.h"

#include <stdex/Memory.h>

#include <sstream>
#include <string>

namespace luci
{

bool is_valid(const circle::OperatorCode *opcode)
{
  circle::BuiltinOperator code = opcode->builtin_code();
  return (circle::BuiltinOperator_MIN <= code && code <= circle::BuiltinOperator_MAX);
}

bool is_custom(const circle::OperatorCode *opcode)
{
  circle::BuiltinOperator code = opcode->builtin_code();
  return (code == circle::BuiltinOperator_CUSTOM);
}

std::string opcode_name(const circle::OperatorCode *opcode)
{
  assert(opcode);

  if (!is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid)";
    return oss.str();
  }

  if (is_custom(opcode))
  {
    if (!opcode->custom_code())
      return "(invalid custom)";

    return opcode->custom_code()->c_str();
  }

  circle::BuiltinOperator code = opcode->builtin_code();
  return circle::EnumNameBuiltinOperator(code);
}

const char *tensor_type(const circle::Tensor *tensor)
{
  return circle::EnumNameTensorType(tensor->type());
}

const char *tensor_name(const circle::Tensor *tensor)
{
  static const char *kEmptyTensorName = "(noname)";

  auto name = tensor->name();
  if (name)
    return name->c_str();

  return kEmptyTensorName;
}

const circle::QuantizationParameters *tensor_quantization(const circle::Tensor *tensor)
{
  return tensor->quantization();
}

loco::DataType luci_datatype(const circle::TensorType type)
{
  switch (type)
  {
    case circle::TensorType_FLOAT32:
      return loco::DataType::FLOAT32;
    case circle::TensorType_FLOAT16:
      return loco::DataType::FLOAT16;
    case circle::TensorType_INT32:
      return loco::DataType::S32;
    case circle::TensorType_UINT8:
      return loco::DataType::U8;
    case circle::TensorType_INT64:
      return loco::DataType::S64;
    case circle::TensorType_STRING:
      break;
    case circle::TensorType_BOOL:
      break;
    case circle::TensorType_INT16:
      return loco::DataType::S16;
    case circle::TensorType_COMPLEX64:
      break;
    case circle::TensorType_INT8:
      return loco::DataType::S8;
  }
  assert(false);
  return loco::DataType::Unknown;
}

loco::DataType luci_datatype(const circle::Tensor *tensor)
{
  // TODO use luci_datatype(circle::TensorType type)
  switch (tensor->type())
  {
    case circle::TensorType_FLOAT32:
      return loco::DataType::FLOAT32;
    case circle::TensorType_FLOAT16:
      return loco::DataType::FLOAT16;
    case circle::TensorType_INT32:
      return loco::DataType::S32;
    case circle::TensorType_UINT8:
      return loco::DataType::U8;
    case circle::TensorType_INT64:
      return loco::DataType::S64;
    case circle::TensorType_STRING:
      break;
    case circle::TensorType_BOOL:
      break;
    case circle::TensorType_INT16:
      return loco::DataType::S16;
    case circle::TensorType_COMPLEX64:
      break;
    case circle::TensorType_INT8:
      return loco::DataType::S8;
  }
  assert(false);
  return loco::DataType::Unknown;
}

FusedActFunc luci_actfunc(const circle::ActivationFunctionType type)
{
  switch (type)
  {
    case circle::ActivationFunctionType::ActivationFunctionType_NONE:
      return luci::FusedActFunc::NONE;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU:
      return luci::FusedActFunc::RELU;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU_N1_TO_1:
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU6:
      return luci::FusedActFunc::RELU6;
    case circle::ActivationFunctionType::ActivationFunctionType_TANH:
      break;
    default:
      break;
  }
  assert(false);
  return luci::FusedActFunc::UNDEFINED;
}

Padding luci_padding(const circle::Padding padding)
{
  switch (padding)
  {
    case circle::Padding::Padding_SAME:
      return Padding::SAME;
    case circle::Padding::Padding_VALID:
      return Padding::VALID;
  }
  assert(false);
  return Padding::UNDEFINED;
}

std::unique_ptr<CircleQuantParam>
luci_quantparam(const circle::QuantizationParameters *quantization)
{
  if ((quantization->min() && quantization->max()) ||
      (quantization->scale() && quantization->zero_point()))
  {
    auto quantparam = stdex::make_unique<CircleQuantParam>();

    if (quantization->min())
      quantparam->min = as_index_vector(quantization->min());
    if (quantization->max())
      quantparam->max = as_index_vector(quantization->max());

    if (quantization->scale())
      quantparam->scale = as_index_vector(quantization->scale());
    if (quantization->zero_point())
      quantparam->zerop = as_index_vector(quantization->zero_point());

    return std::move(quantparam);
  }

  return nullptr;
}

size_t CircleReader::buffer_info(uint32_t buf_idx, const uint8_t **buff_data)
{
  *buff_data = nullptr;

  if (buf_idx == 0)
    return 0;

  if (auto *buffer = (*_buffers)[buf_idx])
  {
    if (auto *array = buffer->data())
    {
      if (size_t size = array->size())
      {
        *buff_data = reinterpret_cast<const uint8_t *>(array->data());
        return size;
      }
    }
  }

  return 0;
}

circle::BuiltinOperator CircleReader::builtin_code(const circle::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const circle::OperatorCode *opcode = _op_codes.at(index);

  return opcode->builtin_code();
}

std::string CircleReader::opcode_name(const circle::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const circle::OperatorCode *opcode = _op_codes.at(index);

  if (!is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid: " << index << ")";
    return oss.str();
  }

  return ::luci::opcode_name(opcode);
}

bool CircleReader::parse(const circle::Model *model)
{
  assert(model != nullptr);

  _model = model;

  _subgraphs = _model->subgraphs();
  _buffers = _model->buffers();

  auto opcodes = _model->operator_codes();
  for (const ::circle::OperatorCode *opcode : *opcodes)
  {
    _op_codes.push_back(opcode);
  }

  return true;
}

bool CircleReader::select_subgraph(uint32_t sgindex)
{
  _tensors = nullptr;
  _operators = nullptr;

  _inputs.clear();
  _outputs.clear();

  if (_subgraphs->Length() <= sgindex)
  {
    assert(false);
    return false;
  }

  const circle::SubGraph *subgraph = (*_subgraphs)[sgindex];

  _tensors = subgraph->tensors();
  _operators = subgraph->operators();

  _inputs = as_index_vector(subgraph->inputs());
  _outputs = as_index_vector(subgraph->outputs());

  return true;
}

} // namespace luci
