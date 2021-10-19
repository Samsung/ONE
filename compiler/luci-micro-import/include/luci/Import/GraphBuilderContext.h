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

#ifndef __LUCI_IMPORT_GRAPH_BUILDER_CONTEXT_H__
#define __LUCI_IMPORT_GRAPH_BUILDER_CONTEXT_H__

#include "CircleReader.h"

#include <luci/IR/CircleNode.h>

#include <loco.h>

#include <map>
#include <set>

namespace luci
{

using TensorIndex = int32_t;

/*
 * @brief  Tensor Index to CircleNode
 *         To find CircleNode from TensorIndex
 */
class IndexNodeFinder
{
public:
  void enroll(TensorIndex idx, CircleNode *node);

  CircleNode *node(TensorIndex idx) const;

private:
  using MapIndexNode_t = std::map<TensorIndex, CircleNode *>;

  MapIndexNode_t _table;
};

/**
 * @brief  Set of Tensor Index of outputs of operators
 *         including graph input nodes
 */
class IndexTensorOutputs
{
public:
  void enroll(TensorIndex idx);

  bool find(TensorIndex idx);

private:
  std::set<TensorIndex> _set;
};

/**
 * @brief Class to store context to build loco graph IR from TensorFlow
 */
class GraphBuilderContext
{
public:
  GraphBuilderContext(loco::Graph *g, CircleReader *reader, IndexNodeFinder *nodefinder,
                      IndexTensorOutputs *tensoroutputs)
    : _g(g), _reader(reader), _indexnodefinder(nodefinder), _indextensoroutputs(tensoroutputs)
  {
    // DO NOTHING
  }

  GraphBuilderContext(const GraphBuilderContext &) = delete;
  GraphBuilderContext(GraphBuilderContext &&) = delete;

public:
  loco::Graph *graph() { return _g; }
  CircleReader *reader() { return _reader; }

  IndexNodeFinder *nodefinder() { return _indexnodefinder; }
  IndexTensorOutputs *tensoroutputs() { return _indextensoroutputs; }

private:
  loco::Graph *_g;
  CircleReader *_reader;
  IndexNodeFinder *_indexnodefinder;
  IndexTensorOutputs *_indextensoroutputs;
};

} // namespace luci

#endif // __LUCI_IMPORT_GRAPH_BUILDER_CONTEXT_H__
