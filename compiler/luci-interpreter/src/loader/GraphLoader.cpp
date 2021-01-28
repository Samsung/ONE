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

#include "loader/GraphLoader.h"

#include "loader/KernelBuilder.h"

#include <loco/IR/Algorithm.h>

namespace luci_interpreter
{
namespace
{

template <typename NodeT> Shape getNodeShape(const NodeT *node)
{
  Shape shape(node->rank());
  for (uint32_t i = 0; i < node->rank(); ++i)
  {
    shape.dim(i) = node->dim(i).value();
  }
  return shape;
}

template <DataType DT> const void *getNodeDataImpl(const luci::CircleConst *node, size_t *data_size)
{
  const size_t element_size = getDataTypeSize(DT);
  const int32_t num_elements = node->size<DT>();

  *data_size = num_elements * element_size;
  if (*data_size > 0)
  {
    // FIXME There is no good way to get the pointer to the data currently.
    return &node->at<DT>(0);
  }
  return nullptr;
}

const void *getNodeData(const luci::CircleConst *node, size_t *data_size)
{
  switch (node->dtype())
  {
    case DataType::U8:
      return getNodeDataImpl<DataType::U8>(node, data_size);
    case DataType::FLOAT32:
      return getNodeDataImpl<DataType::FLOAT32>(node, data_size);
    case DataType::S16:
      return getNodeDataImpl<DataType::S16>(node, data_size);
    case DataType::S32:
      return getNodeDataImpl<DataType::S32>(node, data_size);
    case DataType::S64:
      return getNodeDataImpl<DataType::S64>(node, data_size);
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

bool isExecutableNode(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // These nodes denote inputs / outputs of a graph.
    case luci::CircleOpcode::CIRCLECONST:
    case luci::CircleOpcode::CIRCLECUSTOMOUT:
    case luci::CircleOpcode::CIRCLEINPUT:
    case luci::CircleOpcode::CIRCLEOUTPUT:
    case luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE:
    // The following nodes denote outputs of multiple-output nodes.
    case luci::CircleOpcode::CIRCLEIFOUT:
    case luci::CircleOpcode::CIRCLESPLITOUT:
    case luci::CircleOpcode::CIRCLEUNPACKOUT:
      return false;
    default:
      return true;
  }
}

bool isTensorProducingNode(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // Output nodes do not produce tensors.
    case luci::CircleOpcode::CIRCLEOUTPUT:
    // The following nodes are multiple-output nodes. They do not produce tensors, the tensors
    // are produced by the corresponding *Out nodes instead.
    case luci::CircleOpcode::IF:
    case luci::CircleOpcode::SPLIT:
    case luci::CircleOpcode::UNPACK:
      return false;
    default:
      return true;
  }
}

} // namespace

GraphLoader::GraphLoader(
  const loco::Graph *graph, RuntimeGraph *runtime_graph, RuntimeToIR &runtime_to_ir,
  const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
  std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
  : _graph(graph), _runtime_graph(runtime_graph), _runtime_to_ir(runtime_to_ir),
    _graph_to_runtime_graph(graph_to_runtime_graph), _node_to_tensor(node_to_tensor)
{
}

void GraphLoader::loadTensors()
{
  for (uint32_t i = 0; i < _graph->nodes()->size(); ++i)
  {
    const auto *node = loco::must_cast<const luci::CircleNode *>(_graph->nodes()->at(i));

    if (!isTensorProducingNode(node))
      continue;

    // Only Input and Const nodes have shapes. Shapes of intermediate tensors will be inferred.
    Shape shape{};
    if (const auto *input_node = dynamic_cast<const luci::CircleInput *>(node))
    {
      shape = getNodeShape(input_node);
    }
    else if (const auto *const_node = dynamic_cast<const luci::CircleConst *>(node))
    {
      shape = getNodeShape(const_node);
    }

    AffineQuantization quantization;
    if (node->quantparam() != nullptr)
    {
      const luci::CircleQuantParam *params = node->quantparam();
      assert(params->scale.size() == params->zerop.size());
      quantization.scale.assign(params->scale.cbegin(), params->scale.cend());
      quantization.zero_point.assign(params->zerop.cbegin(), params->zerop.cend());
      quantization.quantized_dimension = params->quantized_dimension;
    }

    auto tensor = std::make_unique<Tensor>(node->dtype(), std::move(shape), std::move(quantization),
                                           node->name());

    if (const auto *const_node = dynamic_cast<const luci::CircleConst *>(node))
    {
      size_t data_size{};
      const void *const_data = getNodeData(const_node, &data_size);
      if (const_data != nullptr)
        tensor->writeData(const_data, data_size);
    }

    _node_to_tensor.emplace(node, tensor.get());
    _runtime_to_ir.tensor_to_node.emplace(tensor.get(), node);

    _runtime_graph->addTensor(std::move(tensor));
  }
}

void GraphLoader::initInputOutputTensors() const
{
  auto input_nodes = loco::input_nodes(_graph);
  std::vector<Tensor *> input_tensors(input_nodes.size());
  for (size_t i = 0; i < input_nodes.size(); ++i)
  {
    input_tensors[i] = _node_to_tensor.at(input_nodes[i]);
  }
  _runtime_graph->setInputTensors(input_tensors);

  auto output_nodes = loco::output_nodes(const_cast<loco::Graph *>(_graph));
  std::vector<Tensor *> output_tensors(output_nodes.size());
  for (size_t i = 0; i < output_nodes.size(); ++i)
  {
    const auto *node = loco::must_cast<const luci::CircleOutput *>(output_nodes[i]);
    output_tensors[i] = _node_to_tensor.at(node->from());
  }
  _runtime_graph->setOutputTensors(output_tensors);
}

void GraphLoader::loadOperators()
{
  KernelBuilder kernel_builder(_graph_to_runtime_graph, _node_to_tensor);

  // Create kernels for executable nodes. This has to be done in execution order.
  for (const loco::Node *loco_node :
       loco::postorder_traversal(loco::output_nodes(const_cast<loco::Graph *>(_graph))))
  {
    const auto *node = loco::must_cast<const luci::CircleNode *>(loco_node);

    if (isExecutableNode(node))
    {
      std::unique_ptr<Kernel> kernel = node->accept(&kernel_builder);
      _runtime_to_ir.kernel_to_node.emplace(kernel.get(), node);
      _runtime_graph->addKernel(std::move(kernel));
    }
  }
}

} // namespace luci_interpreter
