/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "QuantizedModelVerifier.h"

#include "VerifyQuantizedNodeLayerWiseGranularity.h"
#include "VerifyQuantizedNodeChannelWiseGranularity.h"
#include "VerifyQuantizedNodeU8Type.h"
#include "VerifyQuantizedNodeS16Type.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>

namespace luci
{

void QuantizedModelVerifier::verify(loco::Graph *g)
{
  if (_quantized_dtype != Type::U8 && _quantized_dtype != Type::S16)
    throw std::runtime_error("Unsupported quantized dtype");

  if (_granularity != Granularity::ChannelWise && _granularity != Granularity::LayerWise)
    throw std::runtime_error("Unsupported granularity");

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    auto node_name = [&circle_node]() {
      if (circle_node->name().length() == 0)
        return std::string("(noname)");

      return circle_node->name();
    };

    // Verify Type
    if (_quantized_dtype == Type::U8)
    {
      VerifyQuantizedNodeU8Type vt;
      if (!circle_node->accept(&vt))
        throw std::runtime_error("Wrong data type detected in " + node_name());
    }
    else if (_quantized_dtype == Type::S16)
    {
      VerifyQuantizedNodeS16Type vt;
      if (!circle_node->accept(&vt))
        throw std::runtime_error("Wrong data type detected in " + node_name());
    }

    // Verify Granularity
    if (_granularity == Granularity::LayerWise)
    {
      VerifyQuantizedNodeLayerWiseGranularity vg;
      if (!circle_node->accept(&vg))
        throw std::runtime_error("Wrong granularity detected in " + node_name());
    }
    else if (_granularity == Granularity::ChannelWise)
    {
      VerifyQuantizedNodeChannelWiseGranularity vg;
      if (!circle_node->accept(&vg))
        throw std::runtime_error("Wrong granularity detected in " + node_name());
    }
  }
}

} // namespace luci
