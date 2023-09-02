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
#include <mio_tflite2121/Helper.h>

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
  bool ellipsis = amount > 0 && size > 8;
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

template <typename T>
void dump_fbvect(std::ostream &os, const flatbuffers::Vector<T> *fbvect, uint32_t size)
{
  for (uint32_t q = 0; q < size; q++)
  {
    if (q)
      os << ", ";
    os << fbvect->Get(q);
  }
}

template <>
void dump_fbvect(std::ostream &os, const flatbuffers::Vector<uint8_t> *fbvect, uint32_t size)
{
  assert(fbvect);
  for (uint32_t q = 0; q < size; q++)
  {
    if (q)
      os << ", ";
    os << static_cast<uint32_t>(fbvect->Get(q));
  }
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const flatbuffers::Vector<T> *fbvect)
{
  if (fbvect == nullptr)
    return os;

  bool ellipsis = (fbvect->size() > 8);
  auto limit_size = ellipsis ? 8 : fbvect->size();

  if (ellipsis)
  {
    os << "(" << fbvect->size() << ") ";
  }

  dump_fbvect(os, fbvect, limit_size);

  if (ellipsis)
  {
    os << " ... ";
  }

  return os;
}

void dump_sub_graph(std::ostream &os, tflread::Reader &reader)
{
  auto tensors = reader.tensors();
  auto operators = reader.operators();

  // dump operands(tensors)
  os << "Operands: T(subgraph index : tensor index) TYPE (shape) (shape_signature) "
     << "B(buffer index) (variable) OperandName" << std::endl;
  for (uint32_t i = 0; i < tensors->Length(); ++i)
  {
    // TODO refactor to some better structure
    auto tensor = tensors->Get(i);
    std::vector<int32_t> dims = {-1};

    if (tensor->shape())
      dims = tflread::as_index_vector(tensor->shape());

    os << "T(" << reader.subgraph_index() << ":" << i << ") " << mio::tflite::tensor_type(tensor)
       << " ";
    os << "(" << dims << ") ";
    if (tensor->shape_signature())
    {
      std::vector<int32_t> dims_sig = tflread::as_index_vector(tensor->shape_signature());
      os << "(" << dims_sig << ") ";
    }
    os << "B(" << tensor->buffer() << ") ";
    if (tensor->is_variable())
    {
      os << "(variable) ";
    }
    os << mio::tflite::tensor_name(tensor) << std::endl;

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
        {
          os << "zeropt(" << q_params->zero_point() << ") ";
          if (q_params->zero_point()->size() > 1)
            os << std::endl << strqindent;
        }
        os << "quantized_dimension(" << q_params->quantized_dimension() << ")";

        os << std::endl;
      }
    }

    if (const auto &s_params = tensor->sparsity())
    {
      std::string strsparsity = "    Sparsity: ";
      std::string strsindent(strsparsity.size(), ' ');
      os << strsparsity;

      if (s_params->traversal_order())
      {
        os << "traversal_order(" << s_params->traversal_order() << ") ";
        os << std::endl << strsindent;
      }
      if (s_params->block_map())
      {
        os << "block_map(" << s_params->block_map() << ") ";
        os << std::endl << strsindent;
      }
      if (const auto &dim_metadata = s_params->dim_metadata())
      {
        uint32_t idx = 0;
        for (const auto &dm : *dim_metadata)
        {
          std::string strdm = "dim_metadata[" + std::to_string(idx++) + "]: ";
          std::string strdm_indent = strsindent + std::string(strdm.size(), ' ');
          os << strdm;

          os << "format(" << tflite::EnumNameDimensionType(dm->format()) << ") ";
          os << std::endl << strdm_indent;

          os << "dense_size(" << dm->dense_size() << ") ";
          os << std::endl << strdm_indent;

          os << "array_segments_type("
             << tflite::EnumNameSparseIndexVector(dm->array_segments_type()) << ") ";
          os << std::endl << strdm_indent;

          os << "array_segments(";
          switch (dm->array_segments_type())
          {
            case tflite::SparseIndexVector_NONE:
              // DO NOTHING
              break;
            case tflite::SparseIndexVector_Int32Vector:
              os << dm->array_segments_as_Int32Vector()->values();
              break;
            case tflite::SparseIndexVector_Uint16Vector:
              os << dm->array_segments_as_Uint16Vector()->values();
              break;
            case tflite::SparseIndexVector_Uint8Vector:
              os << dm->array_segments_as_Uint8Vector()->values();
              break;
            default:
              throw std::runtime_error("Invalid SparseIndexVector type of array_segments");
          }
          os << ")" << std::endl << strdm_indent;

          os << "array_indices_type(" << tflite::EnumNameSparseIndexVector(dm->array_indices_type())
             << ") ";
          os << std::endl << strdm_indent;

          os << "array_indices(";
          switch (dm->array_indices_type())
          {
            case tflite::SparseIndexVector_NONE:
              // DO NOTHING
              break;
            case tflite::SparseIndexVector_Int32Vector:
              os << dm->array_indices_as_Int32Vector()->values();
              break;
            case tflite::SparseIndexVector_Uint16Vector:
              os << dm->array_indices_as_Uint16Vector()->values();
              break;
            case tflite::SparseIndexVector_Uint8Vector:
              os << dm->array_indices_as_Uint8Vector()->values();
              break;
            default:
              throw std::runtime_error("Invalid SparseIndexVector type of array_indices");
          }
          os << ")" << std::endl << strsindent;
        }
      }
    }
    os << std::endl;
  }

  // dump operators
  os << "Operators: O(subgraph index : operator index) OpCodeName " << std::endl;
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

    os << "O(" << reader.subgraph_index() << ":" << i << ") " << op_name << " ";
    os << std::endl;

    if (auto op_prn = OpPrinterRegistry::get().lookup(builtincode))
    {
      op_prn->options(op, os);
    }

    for (auto input : inputs)
    {
      os << "    I T(" << reader.subgraph_index() << ":" << input << ") ";
      if (input >= 0)
      {
        auto tensor = tensors->Get(input);
        os << mio::tflite::tensor_name(tensor);
      }
      os << std::endl;
    }
    for (auto output : outputs)
    {
      os << "    O T(" << reader.subgraph_index() << ":" << output << ") ";
      if (output >= 0)
      {
        auto tensor = tensors->Get(output);
        os << mio::tflite::tensor_name(tensor);
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
    std::string name = mio::tflite::tensor_name(tensor);
    os << "I T(" << reader.subgraph_index() << ":" << input << ") " << name << std::endl;
  }

  for (const auto output : reader.outputs())
  {
    auto tensor = tensors->Get(output);
    std::string name = mio::tflite::tensor_name(tensor);
    os << "O T(" << reader.subgraph_index() << ":" << output << ") " << name << std::endl;
  }

  os << std::endl;
}

void dump_model(std::ostream &os, const tflite::Model *model)
{
  tflread::Reader reader(model);

  uint32_t num_subgraph = reader.num_subgraph();

  // dump model version
  os << "===================================================================" << std::endl;
  os << "Model version: " << reader.version() << std::endl;
  os << " # sub graphs: " << num_subgraph << std::endl;
  os << std::endl;

  auto opcodes = reader.opcodes();
  auto buffers = reader.buffers();
  auto metadata = reader.metadata();
  auto signaturedefs = reader.signaturedefs();

  // dump operator_codes
  os << "Operator Codes: [order] OpCodeName (OpCode Enum)" << std::endl;
  int32_t opcode_index = 0;
  for (auto opcode : opcodes)
  {
    tflite::BuiltinOperator op_code = opcode->builtin_code();
    tflite::BuiltinOperator dp_code = tflite::BuiltinOperator(opcode->deprecated_builtin_code());

    auto op_name = mio::tflite::opcode_name(opcode);
    auto op_version = opcode->version();

    os << "[" << opcode_index << "] " << op_name << " (code: " << op_code
       << ", dep_code: " << dp_code << ", version: " << op_version << ")" << std::endl;

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

  // dump metadata
  if (metadata != nullptr)
  {
    os << "metadata : B(index) name" << std::endl;
    for (uint32_t i = 0; i < metadata->Length(); ++i)
    {
      os << "B(" << metadata->Get(i)->buffer() << ") " << metadata->Get(i)->name()->c_str()
         << std::endl;
    }
    os << std::endl;
  }

  // dump signaturedef
  if (signaturedefs != nullptr)
  {
    os << "SignatureDef" << std::endl;
    for (uint32_t i = 0; i < signaturedefs->Length(); ++i)
    {
      auto sign_i = signaturedefs->Get(i);
      os << "S(" << i << ") signature_key(" << sign_i->signature_key()->c_str() << "), sub_graph("
         << sign_i->subgraph_index() << ")" << std::endl;

      auto inputs_i = sign_i->inputs();
      for (uint32_t t = 0; t < inputs_i->Length(); ++t)
      {
        auto inputs_i_t = inputs_i->Get(t);
        os << "    I(" << t << ")"
           << " T(" << sign_i->subgraph_index() << ":" << inputs_i_t->tensor_index() << ") "
           << inputs_i_t->name()->c_str() << std::endl;
      }

      auto outputs_i = sign_i->outputs();
      for (uint32_t t = 0; t < outputs_i->Length(); ++t)
      {
        auto outputs_i_t = outputs_i->Get(t);
        os << "    O(" << t << ")"
           << " T(" << sign_i->subgraph_index() << ":" << outputs_i_t->tensor_index() << ") "
           << outputs_i_t->name()->c_str() << std::endl;
      }
    }
    os << std::endl;
  }

  for (uint32_t sg = 0; sg < num_subgraph; ++sg)
  {
    reader.select_subgraph(sg);

    os << "-------------------------------------------------------------------" << std::endl;
    os << "Sub-Graph: #" << sg << " " << reader.subgraph_name() << std::endl;
    os << std::endl;

    dump_sub_graph(os, reader);
  }

  os << "===================================================================" << std::endl;
}

} // namespace tfldump

std::ostream &operator<<(std::ostream &os, const tflite::Model *model)
{
  tfldump::dump_model(os, model);
  return os;
}
