/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "GatherLayer.h"

#include "GGMLHelper.h"
#include "OperationUtils.h"
#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::ggml
{

void Validator::visit(const ir::operation::Gather &node)
{
  using ir::operation::Gather;

  const auto input_index{node.getInputs().at(Gather::Input::INPUT)};
  const auto input_node = &_graph.operands().at(input_index);

  _supported = false;

  if (input_node->typeInfo().type() != ir::DataType::QUANT_GGML_Q4_0)
    return;

  _supported = true;
}

void KernelGenerator::visit(const ir::operation::Gather &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto indices_index{node.getInputs().at(ir::operation::Gather::Input::INDICES)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto indices_tensor = _tensor_reg->getPortableTensor(indices_index);

  const auto rank = _ctx.at(input_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis);

  auto fn = std::make_unique<ops::GatherLayer>();

  fn->configure(input_tensor, indices_tensor, output_tensor, axis, _external_context.get());

  _return_fn = std::move(fn);
}

} // namespace onert::backend::ggml

namespace onert::backend::ggml::ops
{

void GatherLayer::configure(const IPortableTensor *input, const IPortableTensor *indices,
                            IPortableTensor *output, int32_t axis, ExternalContext *ctx)
{
  _input = input;
  _indices = indices;
  _axis = axis;
  _output = output;
  _ctx = ctx;
}

void GatherLayer::runByGGMLQuantInputType()
{
  // Supporting condition
  // Input: rank 2
  // Indice: rank < 4 or rank 4 with dim(0) = 1, INT32
  // Axis: 0
  if (_input->getShape().rank() != 2)
    throw std::runtime_error("Gather: block quantized input tensor must be rank 2");

  if (_indices->getShape().rank() >= 4 &&
      (_indices->getShape().rank() != 4 || _indices->getShape().dim(0) != 1))
    throw std::runtime_error("Gather: invalid indices tensor shape");

  if (_indices->data_type() != ir::DataType::INT32)
    throw std::runtime_error("Gather: indices tensor must be int32 type");

  if (_axis != 0)
    throw std::runtime_error("Gather: axis must be 0");

  // convert tensor
  auto input = getGGMLTensor(_input);
  auto indices = getGGMLTensor(_indices);
  auto output = getGGMLTensor(_output);
  {
    output.op = GGML_OP_GET_ROWS;
    output.src[0] = &input;
    output.src[1] = &indices;
  }
  auto *nodes = &output;

  // create graph
  struct ggml_cgraph graph;
  {
    memset(&graph, 0, sizeof(graph));
    graph.n_nodes = 1;
    graph.nodes = &nodes;
  }

  // get cplan
  auto cplan = ggml_graph_plan(&graph, _ctx->maxNumThreads());
  std::vector<uint8_t> buf(cplan.work_size);
  cplan.work_data = buf.data();

  // compute
  ggml_graph_compute(&graph, &cplan);
}

void GatherLayer::run()
{
  switch (_input->data_type())
  {
    case ir::DataType::QUANT_GGML_Q4_0:
      runByGGMLQuantInputType();
      break;
    default:
      throw std::runtime_error("Gather: unsupported input data type");
  }
}

} // namespace onert::backend::ggml::ops
