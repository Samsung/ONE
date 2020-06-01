/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DepthwiseConv2D.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> DepthwiseConv2DChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_depthwiseconv2d_options());

  auto options = operation.depthwiseconv2d_options();

  auto tflite_padding = as_tflite_padding(options.padding());
  auto tflite_activation = as_tflite_activation(options.activation());

  tflite::DepthwiseConv2DOptionsBuilder options_builder{fbb};
  options_builder.add_padding(tflite_padding);
  options_builder.add_stride_w(options.stride_w());
  options_builder.add_stride_h(options.stride_h());
  options_builder.add_depth_multiplier(options.depth_multiplier());
  options_builder.add_fused_activation_function(tflite_activation);
  options_builder.add_dilation_w_factor(options.dilation_w_factor());
  options_builder.add_dilation_h_factor(options.dilation_h_factor());

  return options_builder.Finish().Union();
}

std::unique_ptr<OpChef>
DepthwiseConv2DChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new DepthwiseConv2DChef{operation}};
}
