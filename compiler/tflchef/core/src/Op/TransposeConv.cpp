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

#include "TransposeConv.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> TransposeConvChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  tflite::TransposeConvOptionsBuilder options_builder{fbb};

  assert(operation.has_transpose_conv_options());

  auto tflite_padding = as_tflite_padding(operation.transpose_conv_options().padding());

  options_builder.add_padding(tflite_padding);

  options_builder.add_stride_h(operation.transpose_conv_options().stride_h());
  options_builder.add_stride_w(operation.transpose_conv_options().stride_w());

  // TODO remove calling has_activation
  auto chef_activation = operation.transpose_conv_options().has_activation()
                           ? operation.transpose_conv_options().activation()
                           : tflchef::NONE;
  auto tflite_activation = as_tflite_activation(chef_activation);
  options_builder.add_fused_activation_function(tflite_activation);

  return options_builder.Finish().Union();
}

std::unique_ptr<OpChef> TransposeConvChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new TransposeConvChef{operation}};
}
