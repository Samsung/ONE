/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "luci/Pass/QuantizeWeightsPass.h"
#include "QuantizeWeights.h"
#include "QuantizationUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/Nodes/CircleConst.h>
#include <luci/Log.h>
#include <loco/IR/TensorShape.h>

#include <iostream>
#include <cmath>
#include <functional>
#include <limits>

namespace luci
{

bool QuantizeWeightsPass::run(loco::Graph *g)
{
  LOGGER(l);
  INFO(l) << "QuantizeWeightsPass Start" << std::endl;

  // Quantize weights
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    QuantizeWeights qw(_ctx->input_model_dtype, _ctx->output_model_dtype, _ctx->granularity);
    circle_node->accept(&qw);
  }

  INFO(l) << "QuantizeWeightsPass End" << std::endl;
  return false; // one time run
}

} // namespace luci
