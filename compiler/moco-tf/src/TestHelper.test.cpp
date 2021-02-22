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

#include "TestHelper.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <cstring>

namespace moco
{
namespace tf
{
namespace test
{

void setup_output_node(loco::Graph *graph, loco::Node *last_node)
{
  // add push as output
  auto push_node = graph->nodes()->create<loco::Push>();
  push_node->from(last_node);

  // set the graph output name and node object
  auto graph_output = graph->outputs()->create();
  graph_output->name("output");
  graph_output->dtype(loco::DataType::FLOAT32);
  loco::link(graph_output, push_node);
}

} // namespace test
} // namespace tf
} // namespace moco

#include <moco/IR/Nodes/TFConst.h>

#include <memory>

#include <gtest/gtest.h>

namespace moco
{
namespace tf
{
namespace test
{

TFNodeBuildTester::TFNodeBuildTester()
{
  _graph = loco::make_graph();
  _tensor_names = std::make_unique<moco::SymbolTable>();
}

void TFNodeBuildTester::inputs(const std::vector<std::string> &names)
{
  for (auto name : names)
  {
    auto input = _graph->nodes()->create<moco::TFConst>();
    moco::TensorName name_01(name, 0);
    _tensor_names->enroll(name_01, input);

    _inputs.push_back(input);
  }
}

void TFNodeBuildTester::output(const char *name) { _output = name; }

moco::TFNode *TFNodeBuildTester::output(void)
{
  assert(_output != nullptr);

  moco::TensorName tname(_output, 0);
  return static_cast<moco::TFNode *>(_tensor_names->node(tname));
}

void TFNodeBuildTester::run(tensorflow::NodeDef &nodedef, moco::GraphBuilder &graphbuilder)
{
  assert(_output != nullptr);

  auto node_defs = std::make_unique<moco::NodeDefTable>();
  auto updates = std::make_unique<moco::UpdateQueue>();

  moco::GraphBuilderContext gb_context(_graph.get(), node_defs.get(), _tensor_names.get(),
                                       updates.get());

  EXPECT_TRUE(graphbuilder.validate(nodedef));
  graphbuilder.build(nodedef, &gb_context);

  for (auto &update : updates->queue())
  {
    update->input(_tensor_names.get());
  }

  auto tfnode = output();
  ASSERT_NE(tfnode, nullptr);

  int idx = 0;
  ASSERT_EQ(tfnode->arity(), _inputs.size());
  for (auto input : _inputs)
  {
    ASSERT_EQ(tfnode->arg(idx++), input);
  }
}

} // namespace test
} // namespace tf
} // namespace moco
