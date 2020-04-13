/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <tfldump/Dump.h>

#include "Read.h"
#include "OpPrinter.h"

#include <ostream>

#include <algorithm> // min
#include <iomanip>   // setfill

namespace tfldump
{

void dump_buffer(std::ostream &os, const uint8_t *buffer, size_t size, size_t amount)
{
  std::ios_base::fmtflags saveflags(os.flags());

  bool second = false;
  bool ellipsis = amount > 0 && size > 4;
  size_t count = ellipsis ? std::min(size, amount) : size;

  for (size_t i = 0; i < count; i++)
  {
    if (second)
    {
      os << " ";
    }

    os << std::showbase << std::setfill('0') << std::setw(2);
    os << std::hex << (uint32_t)buffer[i];

    second = true;
  }
  if (ellipsis)
  {
    os << " ...";
  }

  os.flags(saveflags);
}

void dump_vector(std::ostream &os, const std::vector<int32_t> &vs)
{
  uint32_t seq = 0;
  for (auto &v : vs)
  {
    if (seq)
      os << ", ";
    os << v;
    seq++;
  }
}

std::ostream &operator<<(std::ostream &os, const std::vector<int32_t> &vect)
{
  tfldump::dump_vector(os, vect);
  return os;
}

template <typename T> void dump_fbvect(std::ostream &os, const flatbuffers::Vector<T> *fbvect)
{
  if (fbvect == nullptr)
    return;

  bool ellipsis = (fbvect->size() > 4);
  auto limit_size = ellipsis ? 4 : fbvect->size();

  if (ellipsis)
  {
    os << "(" << fbvect->size() << ") ";
  }
  for (uint32_t q = 0; q < limit_size; q++)
  {
    if (q)
      os << ", ";
    os << fbvect->Get(q);
  }
  if (ellipsis)
  {
    os << " ... ";
  }
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const flatbuffers::Vector<T> *fbvect)
{
  dump_fbvect(os, fbvect);
  return os;
}

void dump_model(std::ostream &os, const tflite::Model *model)
{
  tflread::Reader reader(model);

  assert(reader.num_subgraph() == 1);
  reader.select_subgraph(0);

  auto opcodes = reader.opcodes();
  auto buffers = reader.buffers();
  auto tensors = reader.tensors();
  auto operators = reader.operators();

  // dump operator_codes
  os << "Operator Codes: [order] OpCodeName (OpCode Enum)" << std::endl;
  int32_t opcode_index = 0;
  for (auto opcode : opcodes)
  {
    tflite::BuiltinOperator op_code = opcode->builtin_code();
    auto op_name = tflread::opcode_name(opcode);

    os << "[" << opcode_index << "] " << op_name << " (code: " << op_code << ")" << std::endl;

    opcode_index++;
  }
  os << std::endl;

  // dump buffer
  os << "Buffers: B(index) (length) values, if any" << std::endl;
  for (uint32_t i = 0; i < buffers->Length(); ++i)
  {
    const uint8_t *buff_data;
    size_t size = reader.buffer_info(i, &buff_data);

    os << "B(" << i << ") (" << size << ") ";
    if (buff_data != nullptr)
    {
      dump_buffer(os, buff_data, size, 16);
    }
    os << std::endl;
  }
  os << std::endl;

  // dump operands(tensors)
  os << "Operands: T(tensor index) TYPE (shape) B(buffer index) OperandName" << std::endl;
  for (uint32_t i = 0; i < tensors->Length(); ++i)
  {
    // TODO refactor to some better structure
    auto tensor = tensors->Get(i);
    std::vector<int32_t> dims = {-1};

    if (tensor->shape())
      dims = tflread::as_index_vector(tensor->shape());

    os << "T(" << i << ") " << tflread::tensor_type(tensor) << " ";
    os << "(" << dims << ") ";
    os << "B(" << tensor->buffer() << ") ";
    os << tflread::tensor_name(tensor) << std::endl;

    if (auto q_params = tensor->quantization())
    {
      if ((q_params->min() && q_params->max()) || (q_params->scale() && q_params->zero_point()))
      {
        std::string strquantiz = "    Quantization: ";
        std::string strqindent(strquantiz.size(), ' ');
        os << strquantiz;

        if (q_params->min())
        {
          os << "min(" << q_params->min() << ") ";
          if (q_params->min()->size() > 1)
            os << std::endl << strqindent;
        }
        if (q_params->max())
        {
          os << "max(" << q_params->max() << ") ";
          if (q_params->max()->size() > 1)
            os << std::endl << strqindent;
        }
        if (q_params->scale())
        {
          os << "scale(" << q_params->scale() << ") ";
          if (q_params->scale()->size() > 1)
            os << std::endl << strqindent;
        }
        if (q_params->zero_point())
          os << "zeropt(" << q_params->zero_point() << ") ";

        os << std::endl;
      }
    }
  }
  os << std::endl;

  // dump operators
  os << "Operators: O(operator index) OpCodeName " << std::endl;
  os << "    Option(values) ... <-- depending on OpCode" << std::endl;
  os << "    I T(tensor index) OperandName <-- as input" << std::endl;
  os << "    O T(tensor index) OperandName <-- as output" << std::endl;
  for (uint32_t i = 0; i < operators->Length(); ++i)
  {
    const auto op = operators->Get(i);
    tflite::BuiltinOperator builtincode = reader.builtin_code(op);

    const std::vector<int32_t> &inputs = tflread::as_index_vector(op->inputs());
    const std::vector<int32_t> &outputs = tflread::as_index_vector(op->outputs());
    auto op_name = reader.opcode_name(op);

    os << "O(" << i << ") " << op_name << " ";
    os << std::endl;

    if (auto op_prn = OpPrinterRegistry::get().lookup(builtincode))
    {
      op_prn->options(op, os);
    }

    for (auto input : inputs)
    {
      os << "    I T(" << input << ") ";
      if (input >= 0)
      {
        auto tensor = tensors->Get(input);
        os << tflread::tensor_name(tensor);
      }
      os << std::endl;
    }
    for (auto output : outputs)
    {
      os << "    O T(" << output << ") ";
      if (output >= 0)
      {
        auto tensor = tensors->Get(output);
        os << tflread::tensor_name(tensor);
      }
      os << std::endl;
    }
  }
  os << std::endl;

  // dump network inputs/outputs
  os << "Inputs/Outputs: I(input)/O(output) T(tensor index) OperandName" << std::endl;

  for (const auto input : reader.inputs())
  {
    auto tensor = tensors->Get(input);
    std::string name = tflread::tensor_name(tensor);
    os << "I T(" << input << ") " << name << std::endl;
  }

  for (const auto output : reader.outputs())
  {
    auto tensor = tensors->Get(output);
    std::string name = tflread::tensor_name(tensor);
    os << "O T(" << output << ") " << name << std::endl;
  }
}

} // namespace tfldump

std::ostream &operator<<(std::ostream &os, const tflite::Model *model)
{
  tfldump::dump_model(os, model);
  return os;
}
