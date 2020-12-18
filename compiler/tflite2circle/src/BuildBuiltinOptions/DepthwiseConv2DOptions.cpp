/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DepthwiseConv2DOptions.h"
#include "DataLookup.h"

#include <cassert>

namespace tflite2circle
{

flatbuffers::Offset<circle::DepthwiseConv2DOptions>
build_circle_DepthwiseConv2DOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_DepthwiseConv2DOptions();
  assert(tflite_builtin_options);
  circle::DepthwiseConv2DOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_padding(get_circle_padding(tflite_builtin_options->padding()));
  builtin_options_builder.add_stride_w(tflite_builtin_options->stride_w());
  builtin_options_builder.add_stride_h(tflite_builtin_options->stride_h());
  builtin_options_builder.add_depth_multiplier(tflite_builtin_options->depth_multiplier());
  builtin_options_builder.add_fused_activation_function(
    get_circle_activation_function_type(tflite_builtin_options->fused_activation_function()));
  builtin_options_builder.add_dilation_w_factor(tflite_builtin_options->dilation_w_factor());
  builtin_options_builder.add_dilation_h_factor(tflite_builtin_options->dilation_h_factor());
  return builtin_options_builder.Finish();
}

} // namespace tflite2circle
