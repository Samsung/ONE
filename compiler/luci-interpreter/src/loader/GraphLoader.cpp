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

#include <luci/Plan/CircleNodeExecutionPlan.h>
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
    case DataType::U4:
      return getNodeDataImpl<DataType::U4>(node, data_size);
    case DataType::U8:
      return getNodeDataImpl<DataType::U8>(node, data_size);
    case DataType::FLOAT32:
      return getNodeDataImpl<DataType::FLOAT32>(node, data_size);
    case DataType::S4:
      return getNodeDataImpl<DataType::S4>(node, data_size);
    case DataType::S8:
      return getNodeDataImpl<DataType::S8>(node, data_size);
    case DataType::S16:
      return getNodeDataImpl<DataType::S16>(node, data_size);
    case DataType::S32:
      return getNodeDataImpl<DataType::S32>(node, data_size);
    case DataType::S64:
      return getNodeDataImpl<DataType::S64>(node, data_size);
    case DataType::BOOL:
      return getNodeDataImpl<DataType::BOOL>(node, data_size);
    default:
      throw std::runtime_error("luci-intp (getNodeData) Unsupported type.");
  }
}

const void *getNodeData(const luci::CircleCustom *node, size_t *data_size)
{
  if (node->custom_code() != "CircleReferencingConst")
    return nullptr;

  // helper struct which describes data loaded to custom_options of CircleReferencingConst node
  // TODO move this struct to header
  struct ConstDataReference
  {
    const uint8_t *data = nullptr;
    uint32_t size = 0;
  };

  const auto &custom_options = node->custom_options();
  const auto &const_data_ref = *reinterpret_cast<const ConstDataReference *>(custom_options.data());

  *data_size = const_data_ref.size;
  return const_data_ref.data;
}

bool isExecutableNode(const luci::CircleNode *node)
{
  switch (node->opcode())
  {
    // These nodes denote inputs / outputs of a graph.
    case luci::CircleOpcode::CIRCLECONST:
    case luci::CircleOpcode::CIRCLEINPUT:
    case luci::CircleOpcode::CIRCLEOUTPUT:
    case luci::CircleOpcode::CIRCLEOUTPUTEXCLUDE:
    // The following nodes denote outputs of multiple-output nodes.
    case luci::CircleOpcode::CIRCLEBIDIRECTIONAL_SEQUENCE_LSTM_OUT:
    case luci::CircleOpcode::CIRCLECUSTOMOUT:
    case luci::CircleOpcode::CIRCLEIFOUT:
    case luci::CircleOpcode::CIRCLENONMAXSUPPRESSIONV4OUT:
    case luci::CircleOpcode::CIRCLENONMAXSUPPRESSIONV5OUT:
    case luci::CircleOpcode::CIRCLESPLITOUT:
    case luci::CircleOpcode::CIRCLESPLITVOUT:
    case luci::CircleOpcode::CIRCLETOPKV2OUT:
    case luci::CircleOpcode::CIRCLEUNIQUEOUT:
    case luci::CircleOpcode::CIRCLEUNPACKOUT:
    case luci::CircleOpcode::CIRCLEVARIABLE:
    case luci::CircleOpcode::CIRCLEWHILEOUT:
      return false;
    // Custom nodes may be executable and non-executable
    case luci::CircleOpcode::CUSTOM:
    {
      auto const custom_node = loco::must_cast<const luci::CircleCustom *>(node);

      // TODO handle more non-executable Custom ops here
      if (custom_node->custom_code() == "CircleReferencingConst")
        return false;

      return true;
    }
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
    case luci::CircleOpcode::BIDIRECTIONAL_SEQUENCE_LSTM:
    case luci::CircleOpcode::CUSTOM:
    case luci::CircleOpcode::IF:
    case luci::CircleOpcode::NON_MAX_SUPPRESSION_V4:
    case luci::CircleOpcode::NON_MAX_SUPPRESSION_V5:
    case luci::CircleOpcode::SPLIT:
    case luci::CircleOpcode::SPLIT_V:
    case luci::CircleOpcode::TOPK_V2:
    case luci::CircleOpcode::UNIQUE:
    case luci::CircleOpcode::UNPACK:
    case luci::CircleOpcode::WHILE:
      return false;
    default:
      return true;
  }
}

bool isSupportedCustomNode(const luci::CircleNode *node)
{
  const auto custom_node = loco::must_cast<const luci::CircleCustom *>(node);

  // TODO handle more Custom ops here
  if (custom_node->custom_code() == "CircleReferencingConst")
    return true;

  return false;
}

} // namespace

GraphLoader::GraphLoader(
  const loco::Graph *graph, RuntimeGraph *runtime_graph, RuntimeToIR &runtime_to_ir,
  const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
  std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor, IMemoryManager *memory_manager)
  : _graph(graph), _runtime_graph(runtime_graph), _runtime_to_ir(runtime_to_ir),
    _graph_to_runtime_graph(graph_to_runtime_graph), _node_to_tensor(node_to_tensor),
    _memory_manager(memory_manager)
{
}

void GraphLoader::loadTensors()
{
  for (uint32_t i = 0; i < _graph->nodes()->size(); ++i)
  {
    const auto *node = loco::must_cast<const luci::CircleNode *>(_graph->nodes()->at(i));

    if (node->opcode() == luci::CircleOpcode::CUSTOM && !isSupportedCustomNode(node))
    {
      const auto *cnode = loco::must_cast<const luci::CircleCustom *>(node);
      throw std::runtime_error("Unsupported Custom operator. " + cnode->custom_code() + " in " +
                               node->name());
    }

    if (!isTensorProducingNode(node))
      continue;

    // Only Input, Const, Custom and Variable nodes have shapes. Shapes of intermediate tensors will
    // be inferred.
    Shape shape{};
    switch (node->opcode())
    {
      case luci::CircleOpcode::CIRCLECONST:
      case luci::CircleOpcode::CIRCLECUSTOMOUT:
      case luci::CircleOpcode::CIRCLEINPUT:
      case luci::CircleOpcode::CIRCLEVARIABLE:
        shape = getNodeShape(node);
        break;
      default:
        break;
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

    // If node has execution plan then read memory offsets for nodes
    // from the beginning of shared memory buffer. Used in Static Memory Manager.
    if (luci::has_execution_plan(node))
    {
      auto execution_plan = luci::get_execution_plan(node);
      assert(!execution_plan.offsets().empty());
      tensor->set_offset(execution_plan.offsets().front());
    }

    if (const auto *const_node = dynamic_cast<const luci::CircleConst *>(node))
    {
      size_t data_size{};
      const void *const_data = getNodeData(const_node, &data_size);
      if (const_data != nullptr)
      {
        tensor->set_raw_size(data_size);
        _memory_manager->allocate_memory(*tensor);
        tensor->writeData(const_data, data_size);
      }
      tensor->set_compression(const_node->compression());
    }
    else if (const auto *custom_out_node = dynamic_cast<const luci::CircleCustomOut *>(node))
    {
      const auto *custom_node =
        loco::must_cast<const luci::CircleCustom *>(custom_out_node->input());

      if (custom_node->custom_code() == "CircleReferencingConst")
      {
        size_t data_size{};
        const void *const_data = getNodeData(custom_node, &data_size);
        if (const_data != nullptr)
        {
          tensor->set_raw_size(data_size);
          _memory_manager->allocate_memory(*tensor);
          tensor->writeData(const_data, data_size);
        }
      }
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
    _memory_manager->allocate_memory(*input_tensors[i]);
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
  auto graph = const_cast<loco::Graph *>(_graph);

  auto const graph_nodes = loco::all_nodes(graph);

  // Checking for execution plan in node annotations.
  bool has_execution_annotation = true;
  auto const checking_exec_plan = [&has_execution_annotation](auto const node) {
    const auto *circle_node = loco::must_cast<const luci::CircleNode *>(node);
    if (!luci::has_execution_plan(circle_node))
      has_execution_annotation = false;
  };
  std::for_each(begin(graph_nodes), end(graph_nodes), checking_exec_plan);

  if (has_execution_annotation)
  {
    // Build ordered_nodes vector that stores the order of execution of graph nodes.
    std::vector<const luci::CircleNode *> ordered_nodes(graph_nodes.size());

    auto const filler = [&ordered_nodes](auto const node) {
      const auto *circle_node = loco::must_cast<const luci::CircleNode *>(node);
      auto const position = luci::get_execution_plan(circle_node).order_in_plan();
      ordered_nodes.at(position) = circle_node;
    };
    std::for_each(begin(graph_nodes), end(graph_nodes), filler);

    for (auto node : ordered_nodes)
    {
      if (isExecutableNode(node))
      {
        std::unique_ptr<Kernel> kernel = kernel_builder.build(node);
        _runtime_to_ir.kernel_to_node.emplace(kernel.get(), node);
        _runtime_graph->addKernel(std::move(kernel));
      }
    }
  }
  else
  {
    // If it is impossible to build the execution order plan,
    // then we use the default postorder_traversal approach.
    for (const loco::Node *loco_node : loco::postorder_traversal(loco::output_nodes(graph)))
    {
      const auto *node = loco::must_cast<const luci::CircleNode *>(loco_node);
      if (isExecutableNode(node))
      {
        std::unique_ptr<Kernel> kernel = kernel_builder.build(node);
        _runtime_to_ir.kernel_to_node.emplace(kernel.get(), node);
        _runtime_graph->addKernel(std::move(kernel));
      }
    }
  }
}

} // namespace luci_interpreter
