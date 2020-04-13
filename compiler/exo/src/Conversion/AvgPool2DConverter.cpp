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

#include "AvgPool2DConverter.h"

#include "Dialect/IR/TFLNodes.h"

#include "GraphBlock.h"
#include "Check.h"

#include <loco.h>

namespace exo
{
/**
 * @brief Converts loco::AvgPool2D to locoex::TFLAveragePool2D
 *
 * How it works:  (note: ten->fea means input: tensor, output: feature)
 *
 * Before:
 *   Foo ---- FeatureEncode ---- AvgPool2D ---- FeatureDecode ---- Bar
 *   ten->ten   ten->fea        fea->fea         fea->ten      ten->ten
 *
 * After:              AvgPool2D
 *                    /
 *   Foo -- FeatureEncode - FeatureDecode - TFLAvgPool2D - FeatureEncode - FeatureDecode -- Bar
 *   ten->ten  ten->fea       fea->ten        ten->ten       ten->fea        fea->ten    ten->ten
 *
 * @note  This method replaces AvgPool2D with "FeatureDecode -- TFLAvgPool2D -- FeatureEncode".
 *        Redundant nodes will be removed during transforms.
 */
bool AvgPool2DConverter::convert(loco::AvgPool2D *origin)
{
  auto *graph = origin->graph();

  auto dec = make_feature_decode<FeatureLayout::NHWC>(origin->ifm());
  auto tfl_average = graph->nodes()->create<locoex::TFLAveragePool2D>();
  {
    tfl_average->value(dec);

    // set attributes
    tfl_average->stride()->w(origin->stride()->horizontal());
    tfl_average->stride()->h(origin->stride()->vertical());

    tfl_average->filter()->w(origin->window()->horizontal());
    tfl_average->filter()->h(origin->window()->vertical());

    auto pad = origin->pad();
    if (pad->bottom() == 0 && pad->top() == 0 && pad->left() == 0 && pad->right() == 0)
      tfl_average->padding(locoex::Padding::VALID);
    else
      // TODO This is necessary, but not sufficient condition. More rigorous check required
      tfl_average->padding(locoex::Padding::SAME);

    tfl_average->fusedActivationFunction(locoex::FusedActFunc::NONE);
  }
  auto enc = make_feature_encode<FeatureLayout::NHWC>(tfl_average);

  // replace canonical node
  loco::replace(origin).with(enc);
  origin->ifm(nullptr);

  return true;
}

} // namespace exo
