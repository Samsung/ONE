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

#include "DepthwiseConv2DConverter.h"

#include "Dialect/IR/TFLNodes.h"

#include "GraphBlock.h"
#include "Check.h"

#include <loco.h>
#include <loco/Service/TypeInference.h>
#include <loco/Service/ShapeInference.h>

namespace exo
{

bool DepthwiseConv2DConverter::convert(loco::DepthwiseConv2D *origin)
{
  // Filter shape is required
  if (not loco::shape_known(origin->ker()))
    return false;

  auto filter_shape = loco::shape_get(origin->ker()).as<loco::DepthwiseFilterShape>();

  if ((origin->ifm() == nullptr) or (origin->ker() == nullptr))
    return false;

  auto *graph = origin->graph();

  auto tfl_dw_conv2d = graph->nodes()->create<locoex::TFLDepthwiseConv2D>();
  {
    tfl_dw_conv2d->stride()->w(origin->stride()->horizontal());
    tfl_dw_conv2d->stride()->h(origin->stride()->vertical());

    auto pad = origin->pad();
    if (pad->bottom() == 0 && pad->top() == 0 && pad->left() == 0 && pad->right() == 0)
      tfl_dw_conv2d->padding(locoex::Padding::VALID);
    else
      // TODO This is necessary, but not sufficient condition. More rigorous check required
      tfl_dw_conv2d->padding(locoex::Padding::SAME);

    tfl_dw_conv2d->fusedActivationFunction(locoex::FusedActFunc::NONE);

    uint32_t multiplier = filter_shape.multiplier().value();
    EXO_ASSERT(multiplier < static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
               "Multiplier is too big that casting may occur unintended behavior")

    tfl_dw_conv2d->depthMultiplier(static_cast<int32_t>(multiplier));
  }

  // let's create a new graph connection with tfl_dw_conv2d
  {
    // ifm --- feature_dec --- tfl_dw_conv2d
    auto feature_dec = make_feature_decode<FeatureLayout::NHWC>(origin->ifm());
    tfl_dw_conv2d->input(feature_dec);

    // ker --- filter_dec(H x W x C x M) --- reshape(1 x H x W x CM) --- tfl_dw_conv2d
    auto filter_dec = make_dw_filter_decode<DepthwiseFilterLayout::HWCM>(origin->ker());

    auto reshape = graph->nodes()->create<locoex::TFLReshape>();
    reshape->tensor(filter_dec);

    int32_t new_shape[4] = {
      1, static_cast<int32_t>(filter_shape.height().value()),
      static_cast<int32_t>(filter_shape.width().value()),
      static_cast<int32_t>(filter_shape.depth().value() * filter_shape.multiplier().value())};
    locoex::set_new_shape(reshape, new_shape, 4);

    tfl_dw_conv2d->filter(reshape);

    // bias
    auto zero_const = graph->nodes()->create<locoex::TFLConst>();
    {
      assert(loco::shape_known(origin));
      assert(loco::dtype_known(origin) && loco::dtype_get(origin) == loco::DataType::FLOAT32);

      // bias size is C * M
      uint32_t bias_size = filter_shape.depth().value() * filter_shape.multiplier().value();

      zero_const->dtype(loco::DataType::FLOAT32);
      zero_const->rank(1);
      zero_const->dim(0) = bias_size;
      zero_const->size<loco::DataType::FLOAT32>(bias_size);
      for (uint32_t x = 0; x < bias_size; x++)
        zero_const->at<loco::DataType::FLOAT32>(x) = 0.0;
    }
    tfl_dw_conv2d->bias(zero_const);

    // output
    auto feature_enc = make_feature_encode<FeatureLayout::NHWC>(tfl_dw_conv2d);

    // replace canonical node
    loco::replace(origin).with(feature_enc);
    origin->ifm(nullptr);
  }

  return true;
}

} // namespace exo
