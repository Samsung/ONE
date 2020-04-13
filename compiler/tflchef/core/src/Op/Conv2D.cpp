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

#include "Conv2D.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> Conv2DChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_conv2d_options());

  auto tflite_padding = as_tflite_padding(operation.conv2d_options().padding());
  auto tflite_activation = as_tflite_activation(operation.conv2d_options().activation());

  tflite::Conv2DOptionsBuilder conv2d_options_builder{fbb};
  conv2d_options_builder.add_padding(tflite_padding);
  conv2d_options_builder.add_stride_h(operation.conv2d_options().stride_h());
  conv2d_options_builder.add_stride_w(operation.conv2d_options().stride_w());
  conv2d_options_builder.add_fused_activation_function(tflite_activation);

  return conv2d_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> Conv2DChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new Conv2DChef{operation}};
}
