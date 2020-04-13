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

#include "FeatureBiasAddConverter.h"

#include "Dialect/IR/TFLNodes.h"

#include "GraphBlock.h"

#include <loco.h>
#include <loco/Service/ShapeInference.h>

#include <cassert>

namespace
{

inline void init_fused_act_func(locoex::TFLAdd *node)
{
  node->fusedActivationFunction(locoex::FusedActFunc::NONE);
}

} // namespace

namespace exo
{

/**
 * @brief Converts loco::FeatureBiasAdd to locoex::TFLAdd
 *
 * Before:
 *                  Foo ---+
 *                         |
 *                        loco::FeatureBiasAdd - FeatureDecode - ...
 *                         |
 *      Bar - BiasEncode --+
 *
 * After:
 *
 *              Foo - loco::FeatureDecode --+           loco::FeatureBiasAdd
 *                                          |(x)
 *                                          TFLAdd -- loco::FeatureEncode - FeatureDecode - ...
 *                                          |(y)
 *    Bar - BiasEncode - loco::BiasDecode --+
 */
bool FeatureBiasAddConverter::convert(loco::FeatureBiasAdd *origin)
{
  auto *graph = origin->graph();

  auto tfl_add = graph->nodes()->create<locoex::TFLAdd>();

  // handling input x
  assert(loco::shape_get(origin->value()).domain() == loco::Domain::Feature);

  auto fea_dec = make_feature_decode<FeatureLayout::NHWC>(origin->value());
  tfl_add->x(fea_dec);

  // handling input y
  auto bias_dec = graph->nodes()->create<loco::BiasDecode>();
  assert(bias_dec != nullptr);

  bias_dec->input(origin->bias());

  tfl_add->y(bias_dec);

  // fused activation function
  init_fused_act_func(tfl_add);

  // handling output
  auto fea_enc = make_feature_encode<FeatureLayout::NHWC>(tfl_add);

  loco::replace(origin).with(fea_enc);
  origin->value(nullptr);

  return true;
}

} // namespace exo
