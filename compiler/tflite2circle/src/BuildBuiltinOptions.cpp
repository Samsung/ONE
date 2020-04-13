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

#include <cassert>

#include "DataLookup.h"

namespace tflite2circle
{

flatbuffers::Offset<circle::Conv2DOptions>
build_circle_Conv2DOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_Conv2DOptions();
  assert(tflite_builtin_options);
  circle::Conv2DOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_padding(get_circle_padding(tflite_builtin_options->padding()));
  builtin_options_builder.add_stride_w(tflite_builtin_options->stride_w());
  builtin_options_builder.add_stride_h(tflite_builtin_options->stride_h());
  builtin_options_builder.add_fused_activation_function(
      get_circle_activation_function_type(tflite_builtin_options->fused_activation_function()));
  builtin_options_builder.add_dilation_w_factor(tflite_builtin_options->dilation_w_factor());
  builtin_options_builder.add_dilation_h_factor(tflite_builtin_options->dilation_h_factor());
  return builtin_options_builder.Finish();
}

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

flatbuffers::Offset<circle::Pool2DOptions>
build_circle_Pool2DOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_Pool2DOptions();
  assert(tflite_builtin_options);
  circle::Pool2DOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_padding(get_circle_padding(tflite_builtin_options->padding()));
  builtin_options_builder.add_stride_w(tflite_builtin_options->stride_w());
  builtin_options_builder.add_stride_h(tflite_builtin_options->stride_h());
  builtin_options_builder.add_filter_width(tflite_builtin_options->filter_width());
  builtin_options_builder.add_filter_height(tflite_builtin_options->filter_height());
  builtin_options_builder.add_fused_activation_function(
      get_circle_activation_function_type(tflite_builtin_options->fused_activation_function()));
  return builtin_options_builder.Finish();
}

flatbuffers::Offset<circle::ConcatenationOptions>
build_circle_ConcatenationOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_ConcatenationOptions();
  assert(tflite_builtin_options);
  circle::ConcatenationOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_axis(tflite_builtin_options->axis());
  builtin_options_builder.add_fused_activation_function(
      get_circle_activation_function_type(tflite_builtin_options->fused_activation_function()));
  return builtin_options_builder.Finish();
}

flatbuffers::Offset<circle::AddOptions> build_circle_AddOptions(flatbuffers::FlatBufferBuilder &fb,
                                                                const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_AddOptions();
  assert(tflite_builtin_options);
  circle::AddOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_fused_activation_function(
      get_circle_activation_function_type(tflite_builtin_options->fused_activation_function()));
  return builtin_options_builder.Finish();
}

flatbuffers::Offset<circle::ReshapeOptions>
build_circle_ReshapeOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_ReshapeOptions();
  assert(tflite_builtin_options);
  std::vector<int> new_shape_vec{tflite_builtin_options->new_shape()->begin(),
                                 tflite_builtin_options->new_shape()->end()};
  auto new_shape = fb.CreateVector(new_shape_vec);
  circle::ReshapeOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_new_shape(new_shape);
  return builtin_options_builder.Finish();
}

flatbuffers::Offset<circle::PadOptions> build_circle_PadOptions(flatbuffers::FlatBufferBuilder &fb,
                                                                const tflite::Operator *op)
{
  circle::PadOptionsBuilder builtin_options_builder{fb};
  return builtin_options_builder.Finish();
}

flatbuffers::Offset<circle::SubOptions> build_circle_SubOptions(flatbuffers::FlatBufferBuilder &fb,
                                                                const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_SubOptions();
  assert(tflite_builtin_options);
  circle::SubOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_fused_activation_function(
      get_circle_activation_function_type(tflite_builtin_options->fused_activation_function()));
  return builtin_options_builder.Finish();
}

flatbuffers::Offset<circle::DivOptions> build_circle_DivOptions(flatbuffers::FlatBufferBuilder &fb,
                                                                const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_DivOptions();
  assert(tflite_builtin_options);
  circle::DivOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_fused_activation_function(
      get_circle_activation_function_type(tflite_builtin_options->fused_activation_function()));
  return builtin_options_builder.Finish();
}

flatbuffers::Offset<circle::SoftmaxOptions>
build_circle_SoftmaxOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_SoftmaxOptions();
  assert(tflite_builtin_options);
  circle::SoftmaxOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_beta(tflite_builtin_options->beta());
  return builtin_options_builder.Finish();
}

flatbuffers::Offset<circle::FullyConnectedOptions>
build_circle_FullyConnectedOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_FullyConnectedOptions();
  assert(tflite_builtin_options);
  circle::FullyConnectedOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_fused_activation_function(
      get_circle_activation_function_type(tflite_builtin_options->fused_activation_function()));
  // Get FullyConnectedOptionsWeightsFormat
  auto tflite_weight_format = tflite_builtin_options->weights_format();
  if (tflite_weight_format == tflite::FullyConnectedOptionsWeightsFormat_DEFAULT)
    builtin_options_builder.add_weights_format(circle::FullyConnectedOptionsWeightsFormat_DEFAULT);
  else if (tflite_weight_format == tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8)
    builtin_options_builder.add_weights_format(
        circle::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8);
  return builtin_options_builder.Finish();
}

flatbuffers::Offset<circle::ArgMaxOptions>
build_circle_ArgMaxOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_ArgMaxOptions();
  assert(tflite_builtin_options);
  circle::ArgMaxOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_output_type(
      get_circle_tensortype(tflite_builtin_options->output_type()));
  return builtin_options_builder.Finish();
}

} // namespace tflite2circle
