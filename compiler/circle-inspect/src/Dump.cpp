/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Dump.h"
#include "Reader.h"

#include <ostream>
#include <string>
#include <vector>

namespace circleinspect
{

void DumpOperators::run(std::ostream &os, const circle::Model *model)
{
  circleinspect::Reader reader(model);

  assert(reader.num_subgraph() == 1);
  reader.select_subgraph(0);

  auto ops = reader.operators();

  // dump operators
  for (uint32_t i = 0; i < ops->Length(); ++i)
  {
    const auto op = ops->Get(i);

    auto op_name = reader.opcode_name(op);

    os << op_name << std::endl;
  }
}

} // namespace circleinspect

namespace
{

const circle::Operator *operator_match_output(circleinspect::Reader &reader, const int32_t tensor)
{
  auto ops = reader.operators();

  for (uint32_t i = 0; i < ops->Length(); ++i)
  {
    const auto op = ops->Get(i);

    const std::vector<int32_t> &outputs = circleinspect::as_index_vector(op->outputs());

    for (auto output : outputs)
    {
      if (output == tensor)
        return op;
    }
  }
  return nullptr;
}

size_t tensor_buffer_size(circleinspect::Reader &reader, const int32_t tensor_id)
{
  auto tensors = reader.tensors();

  if (tensor_id < 0 || tensor_id >= tensors->Length())
  {
    throw std::runtime_error("Invalid Tensor ID");
  }

  auto tensor = tensors->Get(tensor_id);
  auto buffer_id = tensor->buffer();

  size_t size = reader.buffer_info(buffer_id, nullptr);

  return size;
}

} // namespace

namespace circleinspect
{

void DumpConv2DWeight::run(std::ostream &os, const circle::Model *model)
{
  circleinspect::Reader reader(model);

  assert(reader.num_subgraph() == 1);
  reader.select_subgraph(0);

  auto ops = reader.operators();

  // dump Conv2D, DepthwiseConv2D and its weight input operator
  for (uint32_t i = 0; i < ops->Length(); ++i)
  {
    const auto op = ops->Get(i);
    auto bc = reader.builtin_code(op);

    if (bc == circle::BuiltinOperator_CONV_2D || bc == circle::BuiltinOperator_DEPTHWISE_CONV_2D)
    {
      const std::vector<int32_t> &inputs = circleinspect::as_index_vector(op->inputs());
      if (inputs.size() < 2)
      {
        throw std::runtime_error("Operator has invalid input");
      }
      auto weight_input = inputs[1]; // Tensor ID of weight input

      const auto op_weight = operator_match_output(reader, weight_input);
      const auto buffer_size = tensor_buffer_size(reader, weight_input);

      std::string weight_op_name = "?";

      if (op_weight == nullptr && buffer_size > 0)
      {
        weight_op_name = "CONST";
      }
      else if (op_weight != nullptr)
      {
        weight_op_name = reader.opcode_name(op_weight);
      }

      auto op_name = reader.opcode_name(op);
      os << op_name << "," << weight_op_name << std::endl;
    }
  }
}

} // namespace circleinspect

namespace
{

template <typename CNTNR>
void print_comma_sepearted(std::ostream &os, const flatbuffers::Vector<CNTNR> *vec)
{
  if (vec == nullptr)
    return;
  for (auto iter = vec->begin(); iter != vec->end(); iter++)
  {
    if (iter != vec->begin())
      os << ", ";
    os << *iter;
  }
}

void print_buffer(std::ostream &os, uint32_t buff_idx, const flatbuffers::Vector<uint8_t> *data_ptr,
                  const circle::TensorType &type)
{
  if (data_ptr == nullptr)
    return;

  os << " └── buffer" << std::endl;
  os << "     ├── index : " << buff_idx << std::endl;
  size_t buff_size = data_ptr->size();
  os << "     ├── size  : " << buff_size << std::endl;
  os << "     └── data  : ";
  switch (type)
  {
    case circle::TensorType_UINT8:
    {
      const uint8_t *buff_data_ui8 = reinterpret_cast<const uint8_t *>(data_ptr->data());
      for (uint32_t idx = 0; idx < buff_size / sizeof(uint8_t); idx++)
      {
        os << static_cast<const uint32_t>(buff_data_ui8[idx]) << ", ";
      }
      break;
    }
    case circle::TensorType_INT32:
    {
      const int32_t *buff_data_i32 = reinterpret_cast<const int32_t *>(data_ptr->data());
      for (uint32_t idx = 0; idx < buff_size / sizeof(int32_t); idx++)
      {
        os << buff_data_i32[idx] << ", ";
      }
      break;
    }
    case circle::TensorType_INT64:
    {
      const int64_t *buff_data_i64 = reinterpret_cast<const int64_t *>(data_ptr->data());
      for (uint32_t idx = 0; idx < buff_size / sizeof(int64_t); idx++)
      {
        os << buff_data_i64[idx] << ", ";
      }
      break;
    }
    case circle::TensorType_FLOAT32:
    {
      const float *buff_data_f32 = reinterpret_cast<const float *>(data_ptr->data());
      for (uint32_t idx = 0; idx < buff_size / sizeof(float); idx++)
      {
        os << buff_data_f32[idx] << ", ";
      }
      break;
    }
    default:
      throw std::runtime_error("NYI tensor type : " + std::to_string(type));
  }
  os << std::endl;
}

} // namepsace

namespace circleinspect
{

void DumpTensors::run(std::ostream &os, const circle::Model *model)
{
  circleinspect::Reader reader(model);
  uint32_t num_subgraph = reader.num_subgraph();
  auto buffers = reader.buffers();

  for (uint32_t subgraph_idx = 0; subgraph_idx < num_subgraph; subgraph_idx++)
  {
    reader.select_subgraph(subgraph_idx);

    auto tensors = reader.tensors();
    for (const auto &tensor : *tensors)
    {
      os << std::string(50, '-') << std::endl;
      os << "[" << tensor->name()->str() << "]" << std::endl;
      auto buff_idx = tensor->buffer();
      auto buff_data_ptr = reader.buffers()->Get(buff_idx)->data();
      auto quant_param = tensor->quantization();
      std::string print_format = (!buff_data_ptr && !quant_param) ? "└──" : "├──";

      // shape
      auto shape = tensor->shape();
      os << " " + print_format + " shape : (";
      ::print_comma_sepearted(os, shape);
      os << ")" << std::endl;

      // quantization paramters
      if (quant_param)
      {
        std::string print_format1 = buff_data_ptr ? "├──" : "└──";
        std::string print_format2 = buff_data_ptr ? "│" : " ";
        os << " " + print_format1 + " quantization" << std::endl;
        auto min = quant_param->min();
        auto max = quant_param->max();
        auto scale = quant_param->scale();
        auto zero_point = quant_param->zero_point();

        os << " " + print_format2 + "   ├── min        : ";
        ::print_comma_sepearted(os, min);
        os << std::endl;
        os << " " + print_format2 + "   ├── max        : ";
        ::print_comma_sepearted(os, max);
        os << std::endl;
        os << " " + print_format2 + "   ├── scale      : ";
        ::print_comma_sepearted(os, scale);
        os << std::endl;
        os << " " + print_format2 + "   └── zero_point : ";
        ::print_comma_sepearted(os, zero_point);
        os << std::endl;
      }

      // buffer
      print_buffer(os, buff_idx, buff_data_ptr, tensor->type());
      os << std::endl;
    }
  }
}

} // namespace circleinspect
