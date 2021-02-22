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

#include "moco/IR/TFNode.h"
#include "moco/IR/TFDialect.h"

#include <memory>
#include <cassert>

namespace moco
{

const loco::Dialect *TFNode::dialect(void) const { return TFDialect::get(); }

} // namespace moco

namespace moco
{

struct GraphInputIndexAnnotation : public loco::NodeAnnotation
{
public:
  GraphInputIndexAnnotation(const loco::GraphInputIndex &index) : _index{index}
  {
    // DO NOTHING
  }

public:
  const loco::GraphInputIndex &index(void) const { return _index; }

private:
  loco::GraphInputIndex _index;
};

bool indexed(const TFPlaceholder *node)
{
  return (node->annot<GraphInputIndexAnnotation>() != nullptr);
}

loco::GraphInputIndex index(const TFPlaceholder *node)
{
  assert(indexed(node));
  return node->annot<GraphInputIndexAnnotation>()->index();
}

void index(TFPlaceholder *node, const loco::GraphInputIndex index)
{
  node->annot(std::make_unique<GraphInputIndexAnnotation>(index));
}

loco::TensorShape tensor_shape(const TFPlaceholder *node)
{
  assert(node != nullptr);

  loco::TensorShape shape;

  uint32_t rank = node->rank();
  shape.rank(rank);
  for (uint32_t index = 0; index < rank; ++index)
  {
    if (node->dim(index).known())
      shape.dim(index) = node->dim(index).value();
    else
      shape.dim(index).unset();
  }

  return shape;
}

TFPlaceholder *placeholder_node(loco::Graph *g, const loco::GraphInputIndex &idx)
{
  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    if (auto tfplaceholder = dynamic_cast<TFPlaceholder *>(g->nodes()->at(n)))
    {
      if (indexed(tfplaceholder) && index(tfplaceholder) == idx)
      {
        return tfplaceholder;
      }
    }
  }
  return nullptr;
}

} // namespace moco

namespace moco
{

/**
 * TFPush
 */

void TFPush::index(const loco::GraphOutputIndex &index)
{
  // Push internally stores "GraphOutputIndex" as int64_t
  _index = static_cast<int64_t>(index);
}

loco::GraphOutputIndex TFPush::index(void) const
{
  assert(_index >= std::numeric_limits<loco::GraphOutputIndex>::min());
  assert(_index <= std::numeric_limits<loco::GraphOutputIndex>::max());
  return static_cast<loco::GraphOutputIndex>(_index);
}

TFPush *push_node(loco::Graph *g, const loco::GraphOutputIndex &index)
{
  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    if (auto tfpush = dynamic_cast<TFPush *>(g->nodes()->at(n)))
    {
      if (tfpush->indexed() && tfpush->index() == index)
      {
        return tfpush;
      }
    }
  }
  return nullptr;
}

} // namespace moco
