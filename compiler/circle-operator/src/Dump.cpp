/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <mio_circle/Helper.h>
#include <mio_circle/Reader.h>

#include <ostream>

namespace
{

// TODO handle multiple outputs case
const circle::Tensor *get_output_tensor(mio::circle::Reader &reader, const circle::Operator *op)
{
  const auto tensors = reader.tensors();
  const auto output_tensors = reader.outputs(op);
  const auto output = output_tensors.at(0);
  return tensors->Get(output);
}

void dump_shape(std::ostream &os, const ::flatbuffers::Vector<int32_t> *shape)
{
  os << "[";
  for (uint32_t i = 0; i < shape->size(); ++i)
  {
    os << shape->Get(i);
    if (i < shape->size() - 1)
    {
      os << ",";
    }
  }
  os << "]";
}

void dump_ops(std::ostream &os, mio::circle::Reader &reader, const cirops::DumpOption &option)
{
  auto ops = reader.operators();
  for (uint32_t i = 0; i < ops->size(); ++i)
  {
    const auto op = ops->Get(i);
    const auto op_name = reader.opcode_name(op);

    if (option.all_graphs)
    {
      // NOTE all_graphs is false for now
      // TODO check using '$' as split key
      os << i << "$";
    }

    if (option.codes)
    {
      const auto op_name = reader.opcode_name(op);
      os << op_name;
    }

    if (option.names)
    {
      const std::string name = mio::circle::tensor_name(get_output_tensor(reader, op));
      os << (option.codes ? "," : "") << name;
    }

    if (option.shapes)
    {
      os << (option.codes || option.names ? "," : "");
      auto const out_tensor = get_output_tensor(reader, op);
      dump_shape(os, (nullptr == out_tensor->shape_signature()) ? out_tensor->shape()
                                                                : out_tensor->shape());
    }
    os << std::endl;
  }
}

} // namespace

namespace cirops
{

void DumpOperators::run(std::ostream &os, const circle::Model *model, const DumpOption &option)
{
  mio::circle::Reader reader(model);

  const uint32_t subgraph_size = reader.num_subgraph();
  for (uint32_t g = 0; g < subgraph_size; g++)
  {
    reader.select_subgraph(g);
    dump_ops(os, reader, option);

    if (!option.all_graphs)
      break;
  }
}

} // namespace cirops
