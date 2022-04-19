/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/UnrepresentableWarningIssuerPass.h"
#include "luci/Pass/QuantizationParameters.h"
#include "helpers/LayerInfoMap.h"

#include <cmath>
#include <luci/IR/CircleNodes.h>
#include <luci/Log.h>

namespace luci
{

void UnrepresentableWarningIssuerPass::warn_if_unrepr(CircleConst *pConst, loco::Graph *g)
{
  LOGGER(l);

  QuantizationGranularity gran = ChannelWise;
  for (auto suc : loco::succs(pConst))
  {
    gran = quantize_granularity(loco::must_cast<const luci::CircleNode *>(suc));
  }

  if (is_unrepr(pConst, _ctx->input_model_dtype, quantize_dtype(pConst), gran))
  {
    WARN(l) << "Weight " << pConst->name()
            << " is poorly representable given quantization parameters: " << std::endl;

    ;
  }
}

bool luci::UnrepresentableWarningIssuerPass::run(loco::Graph *g)
{
  LOGGER(l);
  info_by_name = layer_info_map(g, _ctx->layers_info);

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_const = dynamic_cast<luci::CircleConst *>(node);
    if (not circle_const)
      continue;
    INFO(l) << "UnrepresentableWarningIssuerPass visit node: " << circle_const->name() << std::endl;
    warn_if_unrepr(circle_const, g);
  }

  // One time run
  return false;
}

int nbits(loco::DataType output_t) noexcept
{
  switch (output_t)
  {
    case loco::DataType::S8:
    case loco::DataType::U8:
      return 8;
    case loco::DataType::S16:
    case loco::DataType::U16:
    case loco::DataType::FLOAT16:
      return 16;
    case loco::DataType::S32:
    case loco::DataType::U32:
    case loco::DataType::FLOAT32:
      return 32;
    case loco::DataType::S64:
      return 64;
    default:
      return 64;
  }
}

bool UnrepresentableWarningIssuerPass::is_unrepr(luci::CircleConst *pConst, loco::DataType input_t,
                                                 loco::DataType output_t, QuantizationGranularity)
{
  float min = 1e30f, max = +1e30f;
  float d = 1.5f;
  pConst->rank();
  // TODO only collect min/max across channel for QuantizationGranularity::Channel
  assert(input_t == loco::DataType::FLOAT32);
  assert(pConst->dtype() == loco::DataType::FLOAT32);
  for (uint32_t i = 0; i < pConst->size<loco::DataType::FLOAT32>(); i++)
  {
    auto v = pConst->at<loco::DataType::FLOAT32>(i);
    min = std::min(min, v);
    max = std::max(max, v);
  }
  return log2f(max) - log2f(min) > nbits(output_t) * d;
}

} // namespace luci
