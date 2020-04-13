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

#include "Conv2DConverter.h"

#include "Dialect/IR/TFLNodes.h"

#include "GraphBlock.h"
#include "Check.h"

#include <loco.h>
#include <loco/Service/TypeInference.h>
#include <loco/Service/ShapeInference.h>

namespace exo
{
/**
 * @brief Converts loco::Conv2D to locoex::TFLConv2D
 * @note  Because TFLConv2D accepts input and filter of loco::Domain::Tensor,
 *        loco::FeatureDecode and loco::FilterDecode will be inserted as an inputs
 *        to meet domain invariant.
 *        Please refer to the comment in AvgPool2DConvert.
 */
bool Conv2DConverter::convert(loco::Conv2D *origin)
{
  auto *graph = origin->graph();

  assert(origin->ifm());
  assert(origin->ker());

  auto tfl_conv2d = graph->nodes()->create<locoex::TFLConv2D>();
  {
    tfl_conv2d->stride()->w(origin->stride()->horizontal());
    tfl_conv2d->stride()->h(origin->stride()->vertical());

    auto pad = origin->pad();
    if (pad->bottom() == 0 && pad->top() == 0 && pad->left() == 0 && pad->right() == 0)
      tfl_conv2d->padding(locoex::Padding::VALID);
    else
      // TODO This is necessary, but not sufficient condition. More rigorous check required
      tfl_conv2d->padding(locoex::Padding::SAME);

    tfl_conv2d->fusedActivationFunction(locoex::FusedActFunc::NONE);
  }

  // let's create a new graph connection with tfl_conv2d
  {
    // input
    auto feature_dec = make_feature_decode<FeatureLayout::NHWC>(origin->ifm());
    tfl_conv2d->input(feature_dec);

    // filter
    auto filter_dec = make_filter_decode<FilterLayout::OHWI>(origin->ker());
    tfl_conv2d->filter(filter_dec);

    // bias
    auto zero_const = graph->nodes()->create<locoex::TFLConst>();
    {
      assert(loco::shape_known(origin));
      assert(loco::dtype_known(origin) && loco::dtype_get(origin) == loco::DataType::FLOAT32);

      auto output_depth = loco::shape_get(origin->ker()).as<loco::FilterShape>().count();

      zero_const->dtype(loco::DataType::FLOAT32);
      zero_const->rank(1);
      zero_const->dim(0) = output_depth;
      zero_const->size<loco::DataType::FLOAT32>(output_depth.value());
      for (uint32_t x = 0; x < output_depth.value(); x++)
        zero_const->at<loco::DataType::FLOAT32>(x) = 0.0;
    }
    tfl_conv2d->bias(zero_const);

    // output
    auto feature_enc = make_feature_encode<FeatureLayout::NHWC>(tfl_conv2d);

    // replace canonical node
    loco::replace(origin).with(feature_enc);
    origin->ifm(nullptr);
  }

  return true;
}

} // namespace exo
