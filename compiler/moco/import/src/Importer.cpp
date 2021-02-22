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

#include "moco/Importer.h"
#include "moco/Import/GraphBuilder.h"
#include "moco/Import/GraphBuilderContext.h"

#include "moco/Import/GraphBuilderRegistry.h"

#include <moco/IR/Nodes/TFPlaceholder.h>
#include <moco/IR/TFNode.h>

#include <oops/UserExn.h>

#include <memory>
#include <cassert>
#include <sstream>
#include <stdexcept>

namespace
{

void convert_graph(const moco::GraphBuilderSource &source, const moco::ModelSignature &signature,
                   tensorflow::GraphDef &tf_graph_def, loco::Graph *graph)
{
  auto nodedef = std::make_unique<moco::NodeDefTable>();
  auto tensor_names = std::make_unique<moco::SymbolTable>();
  auto updates = std::make_unique<moco::UpdateQueue>();

  moco::GraphBuilderContext gb_context(graph, nodedef.get(), tensor_names.get(), updates.get());

  // Building a loco graph
  // 1. Convert all the nodes to moco::TFNode
  // 2. Connect inputs: set all node input(from a string) to actual node object
  // 3. Set graph input
  // 4. Create moco::TFPush node and set graph output

  /**
   * @brief Prepare tensorflow::NodeDef search table from name
   */
  for (const auto &n : tf_graph_def.node())
  {
    nodedef->enroll(n.name(), &n);
  }

  /**
   * @brief 1. Convert all the nodes to moco::TFNode
   *
   * @note In each build for a TF node, four things happen
   *       1) create corresponding moco::TFNode(s)
   *       2) read and set the attributes to created moco::TFNode(s)
   *       3) register name-moco::TFNode(last one of Nodes) that will be used as the output
   *       4) queue a task to set the input of the moco::TFNode(first one of the Nodes)
   *          this is done only for required nodes depending on the operator
   *
   * @example Placeholder("in") - Identity("out")
   *        %1 = Placeholder --> 0x1001 (moco::TFNode* object address)
   *        (symboltable: register %1, after the registeration table will contain as below;
   *           "in" : 0x1001
   *        )
   *        (queue: this will be empty as Pull does not queue a task to set input;
   *        )
   *
   *        %2 = Forward --> 0x1002
   *        (symboltable: register %2 and table will look like below;
   *           "in" : 0x1001
   *           "out" : 0x1002
   *        )
   *        (queue: Forward will queue a task with input "in";
   *           0x1002: {"in"}
   *        )
   */
  for (const auto &n : tf_graph_def.node())
  {
    if (const auto *graph_builder = source.lookup(n.op()))
    {
      if (!graph_builder->validate(n))
      {
        throw oops::UserExn("Invalid operator", n.op());
      }

      graph_builder->build(n, &gb_context);
    }
    else
    {
      throw oops::UserExn("Not supported", n.op());
    }
  }

  /**
   * @brief 2. Connect inputs: Iterate updates and call each update input method
   *
   * @note  Continue from above example graph, connecting inputs is done in following steps
   *        a) iterate queue
   *        b) call the input method for each update
   *        c) each update has the moco::TFNode *node and names of the input to connect
   *           node = 0x1002 and names = {"in"}
   *        d) from symbol table, "in" will return 0x1001
   *        e) set input of 0x1002 with 0x1001
   */
  for (auto &update : updates->queue())
  {
    update->input(tensor_names.get());
  }

  /**
   * @brief 3. Set graph input
   */
  for (auto input : signature.inputs())
  {
    auto node = tensor_names->node(input);
    assert(node != nullptr);

    auto graph_input = graph->inputs()->create();

    auto placeholder_node = loco::must_cast<moco::TFPlaceholder *>(node);

    graph_input->name(input.nodeName());

    // annotate index that should be passed to loco::Pull
    moco::index(placeholder_node, graph_input->index());

    // This implementation works as "PlaceholderGraphBuilder in Nodes/Placeholder.cpp"
    // accepts only TF_FLOAT32 as of now.
    //
    // TODO Support other types
    graph_input->dtype(loco::DataType::FLOAT32);
  }

  /**
   * @brief 4. Create moco::TFPush node and set graph output
   */
  for (auto output : signature.outputs())
  {
    auto output_node = tensor_names->node(output);
    assert(output_node);

    // create moco::TFPush for output of graph
    auto push_node = graph->nodes()->create<moco::TFPush>();
    push_node->from(output_node); // set input of TFPush to output node

    // set the graph output name and node object
    auto graph_output = graph->outputs()->create();
    graph_output->name(output.nodeName());
    push_node->index(graph_output->index());

    // TODO Support other types
    graph_output->dtype(loco::DataType::FLOAT32);
  }

  // validate graph
  assert(loco::valid(graph));
}

} // namespace

namespace moco
{

Importer::Importer()
{
  // DO NOTHING
}

std::unique_ptr<loco::Graph> Importer::import(const ModelSignature &signature,
                                              tensorflow::GraphDef &tf_graph_def) const
{
  auto graph = loco::make_graph();

  const GraphBuilderSource *source_ptr = &moco::GraphBuilderRegistry::get();

  if (_source != nullptr)
  {
    // Use user-defined GraphBuilderSource
    source_ptr = _source;
  }

  convert_graph(*source_ptr, signature, tf_graph_def, graph.get());

  return std::move(graph);
}

} // namespace moco
