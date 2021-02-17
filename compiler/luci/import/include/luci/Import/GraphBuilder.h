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

#ifndef __LUCI_IMPORT_GRAPH_BUILDER_H__
#define __LUCI_IMPORT_GRAPH_BUILDER_H__

#include "GraphBuilderContext.h"
#include "GraphBuilderBase.h"

#include <mio/circle/schema_generated.h>

namespace luci
{

/**
 * @brief Base of general single output graph builder(e.g., Conv2DGraphBuilder)
 */
class GraphBuilder : public GraphBuilderBase
{
public:
  virtual ~GraphBuilder() = default;

  // common validate method to check number of inputs and single output
  bool validate(const ValidateArgs &args, size_t i) const
  {
    return (args.op.inputs.size() == i && args.op.outputs.size() == 1);
  }

  CircleNode *build(const circle::OperatorT &op, GraphBuilderContext *context) const final;

private:
  virtual CircleNode *build_node(const circle::OperatorT &op,
                                 const std::vector<CircleNode *> &inputs,
                                 loco::Graph *graph) const = 0;
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPH_BUILDER_H__
