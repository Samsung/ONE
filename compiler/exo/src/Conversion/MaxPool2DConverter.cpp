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

#include "MaxPool2DConverter.h"

#include "Dialect/IR/TFLNodes.h"
#include "GraphBlock.h"

#include <loco.h>

namespace exo
{

/**
 * @brief Converts loco::MaxPool2D to locoex::TFLMaxPool2D
 *
 * @note  This works similar to AvgPool2DConverter. Please refer to the comment in
 *        AvgPool2DConverter.
 */
bool MaxPool2DConverter::convert(loco::MaxPool2D *origin)
{
  auto *graph = origin->graph();

  auto dec = make_feature_decode<FeatureLayout::NHWC>(origin->ifm());
  auto tfl_max = graph->nodes()->create<locoex::TFLMaxPool2D>();
  {
    tfl_max->value(dec);

    // set attributes
    tfl_max->stride()->w(origin->stride()->horizontal());
    tfl_max->stride()->h(origin->stride()->vertical());

    tfl_max->filter()->w(origin->window()->horizontal());
    tfl_max->filter()->h(origin->window()->vertical());

    auto pad = origin->pad();
    if (pad->bottom() == 0 && pad->top() == 0 && pad->left() == 0 && pad->right() == 0)
      tfl_max->padding(locoex::Padding::VALID);
    else
      // TODO This is necessary, but not sufficient condition. More rigorous check required
      tfl_max->padding(locoex::Padding::SAME);

    tfl_max->fusedActivationFunction(locoex::FusedActFunc::NONE);
  }

  auto enc = make_feature_encode<FeatureLayout::NHWC>(tfl_max);

  loco::replace(origin).with(enc);
  origin->ifm(nullptr);

  return true;
}

} // namespace exo
