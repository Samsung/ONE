/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IMPORT_GRAPH_BUILDER_MULTI_OUTPUT_H__
#define __LUCI_IMPORT_GRAPH_BUILDER_MULTI_OUTPUT_H__

#include "GraphBuilderContext.h"
#include "GraphBuilderBase.h"

#include <mio/circle/schema_generated.h>

namespace luci
{

/**
 * @brief Base of general multiple outputs graph builder(e.g., CircleIfGraphBuilder)
 */
class GraphBuilderMultiOutput : public GraphBuilderBase
{
public:
  virtual ~GraphBuilderMultiOutput() = default;

  CircleNode *build(const circle::OperatorT &op, GraphBuilderContext *context) const final;

protected:
  struct BuildNodeArgs
  {
    BuildNodeArgs(const circle::OperatorT &o, GraphBuilderContext *c,
                  const std::vector<CircleNode *> &i)
      : op(o), context(c), input_nodes(i)
    {
    }

    const circle::OperatorT &op;
    GraphBuilderContext *context;
    const std::vector<CircleNode *> &input_nodes;
  };

  struct BuildOutArgs
  {
    BuildOutArgs(CircleNode *nd, uint32_t n) : node(nd), index(n) {}

    CircleNode *node;
    uint32_t index;
  };

private:
  virtual CircleNode *build_node(const BuildNodeArgs &) const = 0;
  virtual CircleNode *build_out(const BuildOutArgs &) const = 0;
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPH_BUILDER_MULTI_OUTPUT_H__
