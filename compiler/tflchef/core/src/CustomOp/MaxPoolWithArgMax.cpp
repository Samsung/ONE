/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#include "MaxPoolWithArgMax.h"

#include "flatbuffers/flexbuffers.h"

flatbuffers::Offset<void> MaxPoolWithArgMaxChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  return flatbuffers::Offset<void>();
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
MaxPoolWithArgMaxChef::custom_value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.type() == "MaxPoolWithArgMax");

  auto flex_buffers = std::make_unique<flexbuffers::Builder>();
  size_t map_start = flex_buffers->StartMap();

  // TODO Support more data types
  flex_buffers->Int("filter_width", operation.max_pool_with_argmax_options().filter_width());
  flex_buffers->Int("filter_height", operation.max_pool_with_argmax_options().filter_height());
  flex_buffers->Int("stride_w", operation.max_pool_with_argmax_options().stride_w());
  flex_buffers->Int("stride_h", operation.max_pool_with_argmax_options().stride_h());
  flex_buffers->Int("padding", operation.max_pool_with_argmax_options().padding());
  flex_buffers->Int("activation", operation.max_pool_with_argmax_options().activation());
  flex_buffers->Int("output_type", operation.max_pool_with_argmax_options().output_type());

  flex_buffers->EndMap(map_start);
  flex_buffers->Finish();

  auto circle_custom_options = fbb.CreateVector(flex_buffers->GetBuffer());
  return circle_custom_options;
}

std::unique_ptr<OpChef>
MaxPoolWithArgMaxChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new MaxPoolWithArgMaxChef{operation}};
}
