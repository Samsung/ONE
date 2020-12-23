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

#include "luci_interpreter/Interpreter.h"

#include "TensorMap.h"
#include "KernelBuilder.h"

#include <loco/IR/Algorithm.h>

#include <stdexcept>
#include <log/luci/Log.h>
namespace luci_interpreter
{

  template <typename NodeT>
  static Shape getNodeShape(const NodeT *node)
  {
    Shape shape(node->rank());
    for (uint32_t i = 0; i < node->rank(); ++i)
    {
      shape.dim(i) = node->dim(i).value();
    }
    return shape;
  }

  template <DataType DT>
  static const void *getNodeDataImpl(const luci::CircleConst *node, size_t *data_size)
  {
    const size_t element_size = getDataTypeSize(DT);
    const int32_t num_elements = node->size<DT>();

    *data_size = num_elements * element_size;
    // FIXME There is no good way to get the pointer to the data currently.
    return &node->at<DT>(0);
  }

  static const void *getNodeData(const luci::CircleConst *node, size_t *data_size)
  {
    switch (node->dtype())
    {
    case DataType::U8:
      return getNodeDataImpl<DataType::U8>(node, data_size);
    case DataType::FLOAT32:
      return getNodeDataImpl<DataType::FLOAT32>(node, data_size);
    case DataType::S32:
      return getNodeDataImpl<DataType::S32>(node, data_size);
    default:
      throw std::runtime_error("Unsupported type.");
    }
  }

  void Interpreter::createTensors(const loco::Graph *graph)
  {
    for (uint32_t i = 0; i < graph->nodes()->size(); ++i)
    {
      const auto *node = dynamic_cast<const luci::CircleNode *>(graph->nodes()->at(i));
      assert(node != nullptr);

      // Output nodes do not produce new tensors.
      if (node->opcode() == luci::CircleOpcode::CIRCLEOUTPUT)
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
        quantization.scale.assign(params->scale.cbegin(), params->scale.cend());
        quantization.zero_point.assign(params->zerop.cbegin(), params->zerop.cend());
      }

      auto tensor = std::make_unique<Tensor>(node->dtype(), std::move(shape), std::move(quantization),
                                             node->name());

      if (const auto *const_node = dynamic_cast<const luci::CircleConst *>(node))
      {
        size_t data_size{};
        const void *const_data = getNodeData(const_node, &data_size);
        tensor->writeData(const_data, data_size);
      }

      _tensor_map->setTensor(node, std::move(tensor));
    }
  }

  void Interpreter::createExecutionSequence(const loco::Graph *main_graph)
  {
    KernelBuilder kernel_builder(*_tensor_map);

    auto nodes = loco::postorder_traversal(loco::output_nodes(const_cast<loco::Graph *>(main_graph)));
    for (loco::Node *loco_node : nodes)
    {

      const auto *node = dynamic_cast<const luci::CircleNode *>(loco_node);
      INFO() << "node->opcode() " << (int)node->opcode() << std::endl;

      assert(node != nullptr);

      if (node->opcode() == luci::CircleOpcode::CONST ||
          node->opcode() == luci::CircleOpcode::CIRCLEINPUT ||
          node->opcode() == luci::CircleOpcode::CIRCLEOUTPUT)
      {
        continue;
      }
      INFO() << "_execution_sequence.push_back node->opcode() " << (int)(node->opcode()) << std::endl;

      _execution_sequence.push_back(node->accept(&kernel_builder));
    }
  }

  Interpreter::Interpreter(const luci::Module *module)
  {
    INFO() << "Interpreter::Interpreter(const luci::Module *module)\n";

    if (module->size() > 1)
    {
      throw std::runtime_error("Models with multiple subgraphs are not yet supported.");
    }
    INFO() << "module->size() " << module->size() << std::endl;

    loco::Graph *main_graph = module->graph();

    _tensor_map = std::make_unique<TensorMap>();
    INFO() << "createTensors(main_graph);\n";
    createTensors(main_graph);
    INFO() << "createExecutionSequence(main_graph);\n";

    createExecutionSequence(main_graph);

    for (const auto &kernel : _execution_sequence)
    {
      INFO() << "kernel->configure();\n";

      kernel->configure();
    }
  }

  Interpreter::~Interpreter() = default;

  void Interpreter::writeInputTensor(const luci::CircleInput *input_node, const void *data,
                                     size_t data_size)
  {
    Tensor *tensor = _tensor_map->getTensor(input_node);
    if (tensor == nullptr)
    {
      const std::string &name = input_node->name();
      throw std::runtime_error("Cannot find tensor for input node named \"" + name + "\".");
    }
    tensor->writeData(data, data_size);
  }

  void Interpreter::readOutputTensor(const luci::CircleOutput *output_node, void *data,
                                     size_t data_size)
  {
    Tensor *tensor = _tensor_map->getTensor(output_node->from());
    if (tensor == nullptr)
    {
      const std::string &name = output_node->name();
      throw std::runtime_error("Cannot find tensor for output node named \"" + name + "\".");
    }
    tensor->readData(data, data_size);
  }

  void Interpreter::interpret()
  {
    for (const auto &kernel : _execution_sequence)
    {
      kernel->execute();
    }
  }

} // namespace luci_interpreter
