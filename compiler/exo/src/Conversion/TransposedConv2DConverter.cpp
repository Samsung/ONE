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

#include "TransposedConv2DConverter.h"

#include "Dialect/IR/TFLNodes.h"

#include "GraphBlock.h"

#include <loco.h>
#include <loco/Service/ShapeInference.h>

namespace exo
{

bool TransposedConv2DConverter::convert(loco::TransposedConv2D *origin)
{
  // Shape is required to set origin->inputSizes()
  if (not loco::shape_known(origin))
    return false;

  if ((origin->ifm() == nullptr) or (origin->ker() == nullptr))
    return false;

  auto *graph = origin->graph();

  auto tfl_tr_conv = graph->nodes()->create<locoex::TFLTransposeConv>();
  {
    tfl_tr_conv->stride()->w(origin->stride()->horizontal());
    tfl_tr_conv->stride()->h(origin->stride()->vertical());

    auto pad = origin->pad();
    if (pad->bottom() == 0 && pad->top() == 0 && pad->left() == 0 && pad->right() == 0)
      tfl_tr_conv->padding(locoex::Padding::VALID);
    else
      // TODO This is necessary, but not sufficient condition. More rigorous check required
      tfl_tr_conv->padding(locoex::Padding::SAME);
  }

  // let's create a new graph connection with tfl_tr_conv
  {
    // Make inputSizes from shape of origin
    auto input_sizes_const = graph->nodes()->create<locoex::TFLConst>();
    auto origin_shape = loco::shape_get(origin).as<loco::FeatureShape>();

    const loco::DataType S32 = loco::DataType::S32;

    input_sizes_const->dtype(S32);
    input_sizes_const->rank(1);
    input_sizes_const->dim(0) = 4;
    input_sizes_const->size<S32>(4);
    // Note that NHWC is layout for inputSizes determined by tflite format
    input_sizes_const->at<S32>(0) = origin_shape.count().value();  // N
    input_sizes_const->at<S32>(1) = origin_shape.height().value(); // H
    input_sizes_const->at<S32>(2) = origin_shape.width().value();  // W
    input_sizes_const->at<S32>(3) = origin_shape.depth().value();  // C

    tfl_tr_conv->inputSizes(input_sizes_const);

    // filter
    auto filter_dec = make_filter_decode<FilterLayout::OHWI>(origin->ker());
    tfl_tr_conv->filter(filter_dec);

    // outBackprop
    auto feature_dec = make_feature_decode<FeatureLayout::NHWC>(origin->ifm());
    tfl_tr_conv->outBackprop(feature_dec);

    // output
    auto feature_enc = make_feature_encode<FeatureLayout::NHWC>(tfl_tr_conv);

    // replace canonical node
    loco::replace(origin).with(feature_enc);
    origin->ifm(nullptr);
  }

  return true;
}

} // namespace exo
