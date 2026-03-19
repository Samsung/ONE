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

#ifndef __TEST_HELPER_H__
#define __TEST_HELPER_H__

#include "moco/Import/GraphBuilder.h"

#include <moco/IR/TFNode.h>
#include <loco.h>
#include <plier/tf/TestHelper.h>

#include <tensorflow/core/framework/graph.pb.h>

#define STRING_CONTENT(content) #content

namespace moco
{
namespace test
{

template <typename T> T *find_first_node_bytype(loco::Graph *g)
{
  T *first_node = nullptr;
  loco::Graph::NodeContext *nodes = g->nodes();
  uint32_t count = nodes->size();

  for (uint32_t i = 0; i < count; ++i)
  {
    first_node = dynamic_cast<T *>(nodes->at(i));
    if (first_node != nullptr)
      break;
  }

  return first_node;
}

} // namespace test
} // namespace moco

namespace moco
{
namespace test
{

class TFNodeBuildTester
{
public:
  TFNodeBuildTester();

public:
  void inputs(const std::vector<std::string> &names);
  void inputs(const std::vector<std::string> &names, const loco::DataType dtype);
  void output(const char *name);
  moco::TFNode *output(void);

  void run(tensorflow::NodeDef &node_def, moco::GraphBuilder &graph_builder);

private:
  std::unique_ptr<moco::SymbolTable> _tensor_names;
  std::unique_ptr<loco::Graph> _graph;

  std::vector<moco::TFNode *> _inputs;
  const char *_output{nullptr};
};

} // namespace test
} // namespace moco

#endif // __TEST_HELPER_H__
