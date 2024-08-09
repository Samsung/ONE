/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mio_circle/Reader.h"
#include "mio_circle/Helper.h"

#include <sstream>
#include <string>

namespace mio
{
namespace circle
{

Reader::Reader(const ::circle::Model *model)
{
  if (model == nullptr)
  {
    throw std::runtime_error("Invalid model");
  }

  _version = model->version();
  _subgraphs = model->subgraphs();
  _buffers = model->buffers();
  _metadata = model->metadata();
  _signature_defs = model->signature_defs();

  auto opcodes = model->operator_codes();
  for (const ::circle::OperatorCode *opcode : *opcodes)
  {
    _op_codes.push_back(opcode);
  }
}

size_t Reader::buffer_info(uint32_t buf_idx, const uint8_t **buff_data)
{
  if (buff_data != nullptr)
  {
    *buff_data = nullptr;
  }

  if (buf_idx == 0)
    return 0;

  if (auto *buffer = (*_buffers)[buf_idx])
  {
    if (auto *array = buffer->data())
    {
      if (size_t size = array->size())
      {
        if (buff_data != nullptr)
        {
          *buff_data = reinterpret_cast<const uint8_t *>(array->data());
        }
        return size;
      }
    }
  }

  return 0;
}

::circle::BuiltinOperator Reader::builtin_code(const ::circle::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const ::circle::OperatorCode *opcode = _op_codes.at(index);

  return mio::circle::builtin_code_neutral(opcode);
}

std::string Reader::opcode_name(const ::circle::Operator *op) const
{
  uint32_t index = op->opcode_index();
  assert(index < _op_codes.size());
  const ::circle::OperatorCode *opcode = _op_codes.at(index);

  if (!mio::circle::is_valid(opcode))
  {
    std::ostringstream oss;
    oss << "(invalid: " << index << ")";
    return oss.str();
  }

  return mio::circle::opcode_name(opcode);
}

std::vector<int32_t> Reader::outputs(const ::circle::Operator *op) const
{
  return as_index_vector(op->outputs());
}

std::string Reader::tensor_name(const ::circle::Tensor *tensor) const
{
  return mio::circle::tensor_name(tensor);
}

std::string Reader::tensor_dtype(const ::circle::Tensor *tensor) const
{
  return mio::circle::tensor_type(tensor);
}

bool Reader::select_subgraph(uint32_t sgindex)
{
  _subgraph_index = sgindex;
  _tensors = nullptr;
  _operators = nullptr;

  _inputs.clear();
  _outputs.clear();

  if (_subgraphs->size() <= sgindex)
  {
    assert(false);
    return false;
  }

  const ::circle::SubGraph *subgraph = (*_subgraphs)[sgindex];

  auto name = subgraph->name();
  _subgraph_name = name ? name->c_str() : "(noname)";

  _tensors = subgraph->tensors();
  _operators = subgraph->operators();

  _inputs = as_index_vector(subgraph->inputs());
  _outputs = as_index_vector(subgraph->outputs());

  return true;
}

} // namespace circle
} // namespace mio
