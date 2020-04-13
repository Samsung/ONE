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

#include "FuseRsqrtPass.h"

#include "Check.h"

#include "Dialect/IR/TFLNodes.h"

namespace
{

/**
 * @return  Casted TFLDiv for fusable candidate, nullptr otherwise
 *
 * This helper checkes fusability with following conditions:
 * - TFLDiv has no activation
 * - TFLDiv's first argument is TFLConst with all value 1
 * - TFLDiv's second argument is TFLSqrt
 */
locoex::TFLDiv *as_candidate(loco::Node *node)
{
  auto div = dynamic_cast<locoex::TFLDiv *>(node);
  if (not div)
    return nullptr;

  // Cannot fuse Div with activation function
  if (div->fusedActivationFunction() != locoex::FusedActFunc::NONE)
    return nullptr;

  auto const_one = dynamic_cast<locoex::TFLConst *>(div->x());
  if (not const_one)
    return nullptr;

  const loco::DataType FLOAT32 = loco::DataType::FLOAT32;
  // TODO Support other dtype
  EXO_ASSERT(const_one->dtype() == FLOAT32, "Only support FLOAT32 now");
  for (uint32_t i = 0; i < const_one->size<FLOAT32>(); ++i)
    if (const_one->at<FLOAT32>(i) != 1.0f)
      return nullptr;

  auto sqrt = dynamic_cast<locoex::TFLSqrt *>(div->y());
  if (not sqrt)
    return nullptr;

  return div;
}

void fuse_rsqrt(locoex::TFLDiv *div)
{
  auto sqrt = dynamic_cast<locoex::TFLSqrt *>(div->y());
  EXO_ASSERT(sqrt, "sqrt should be valid at this point");

  // TFLRsqrt to replace
  auto rsqrt = div->graph()->nodes()->create<locoex::TFLRsqrt>();
  rsqrt->x(sqrt->x());

  // replace
  loco::replace(div).with(rsqrt);
}

} // namespace

namespace exo
{

bool FuseRsqrtPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto div = as_candidate(node))
    {
      fuse_rsqrt(div);
      changed = true;
    }
  }

  return changed;
}

} // namespace exo
