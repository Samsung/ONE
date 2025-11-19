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

#include "FullyConnectedLayer.h"

#include "GGMLHelper.h"
#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::ggml
{

void Validator::visit(const ir::operation::FullyConnected &node)
{
  using ir::operation::FullyConnected;

  const auto weight_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};
  const auto weight_node = &_graph.operands().at(weight_index);

  _supported = false;

  if (weight_node->typeInfo().type() != ir::DataType::QUANT_GGML_Q4_0 &&
      weight_node->typeInfo().type() != ir::DataType::QUANT_GGML_Q8_0)
    return;

  if (node.param().activation != ir::Activation::NONE)
    return;

  _supported = true;
}

void KernelGenerator::visit(const ir::operation::FullyConnected &node)
{
  using ir::operation::FullyConnected;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(FullyConnected::Input::INPUT)};
  const auto weight_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};
  const auto bias_index{node.getInputs().at(FullyConnected::Input::BIAS)};
  const auto activation = node.param().activation;
  const auto weights_format = node.param().weights_format;
  if (weights_format != ir::FullyConnectedWeightsFormat::Default)
    throw std::runtime_error("Unsupported FullyConnected Weights Format");

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto weight_tensor = _tensor_reg->getPortableTensor(weight_index);
  auto bias_tensor = bias_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(bias_index);

  auto fn = std::make_unique<ops::FullyConnectedLayer>();

  fn->configure(input_tensor, weight_tensor, bias_tensor, activation, output_tensor,
                _external_context);

  _return_fn = std::move(fn);
}

} // namespace onert::backend::ggml

namespace onert::backend::ggml::ops
{

FullyConnectedLayer::FullyConnectedLayer()
  : _input(nullptr), _weights(nullptr), _bias(nullptr), _output(nullptr),
    _activation(ir::Activation::NONE), _external_context(nullptr)
{
  // DO NOTHING
}

FullyConnectedLayer::~FullyConnectedLayer() = default;

void FullyConnectedLayer::fullyConnectedGGMLWeight()
{
  if (_bias)
    throw std::runtime_error{"FullyConnected: GGML weights format does not support bias yet."};

  // convert tensor
  auto input = getGGMLTensor(_input);
  auto weights = getGGMLTensor(_weights);
  auto output = getGGMLTensor(_output);
  {
    output.op = GGML_OP_MUL_MAT;
    output.src[0] = &weights;
    output.src[1] = &input;
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
  auto cplan = ggml_graph_plan(&graph, _external_context->maxNumThreads());
  std::vector<uint8_t> buf(cplan.work_size);
  cplan.work_data = buf.data();

  // compute
  ggml_graph_compute(&graph, &cplan);
}

void FullyConnectedLayer::configure(const IPortableTensor *input, const IPortableTensor *weights,
                                    const IPortableTensor *bias, ir::Activation activation,
                                    IPortableTensor *output,
                                    const std::shared_ptr<ExternalContext> &external_context)
{
  _input = input;
  _weights = weights;
  _bias = bias;
  _activation = activation;
  _output = output;
  _external_context = external_context;
}

void FullyConnectedLayer::run()
{
  if (_weights->data_type() == ir::DataType::QUANT_GGML_Q4_0 ||
      _weights->data_type() == ir::DataType::QUANT_GGML_Q8_0)
  {
    fullyConnectedGGMLWeight();
  }
  else
  {
    throw std::runtime_error{"FullyConnected: unsupported data type"};
  }
}

void FullyConnectedLayer::prepare()
{
  // DO NOTHING
}

} // namespace onert::backend::ggml::ops
