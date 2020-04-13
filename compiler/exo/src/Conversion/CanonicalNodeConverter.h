/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __CONVERSION_CANONICAL_NODE_CONVERTER_H__
#define __CONVERSION_CANONICAL_NODE_CONVERTER_H__

#include "Convert.h"

#include <loco.h>
#include <loco/IR/CanonicalDialect.h>
#include <logo/Pass.h>

namespace exo
{

/**
 * @brief Class to convert a canonical node to TFL node
 *
 * TODO Find a better name
 */
template <typename CanonicalType> class CanonicalNodeConverter : public logo::Pass
{
public:
  virtual const char *name(void) const { return nullptr; }

public:
  bool run(loco::Graph *graph);

protected:
  virtual bool convert(CanonicalType *node) = 0;
};

template <typename CanonicalType>
bool CanonicalNodeConverter<CanonicalType>::run(loco::Graph *graph)
{
  auto active_nodes = loco::active_nodes(loco::output_nodes(graph));
  bool changed = false;

  for (auto node : active_nodes)
  {
    // TODO Generalize this to all loco dialects
    if (node->dialect() == loco::CanonicalDialect::get())
    {
      auto the_node = dynamic_cast<CanonicalType *>(node);
      if (the_node != nullptr)
      {
        if (convert(the_node))
          changed = true;
      }
    }
  }

  return changed;
}

} // namespace exo

#endif //__CONVERSION_CANONICAL_NODE_CONVERTER_H__
