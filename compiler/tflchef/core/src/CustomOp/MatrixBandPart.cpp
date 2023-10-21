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

#include "MatrixBandPart.h"
#include "OpUtils.h"

#include <flatbuffers/flexbuffers.h>

flatbuffers::Offset<void> MatrixBandPartChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  return flatbuffers::Offset<void>();
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
MatrixBandPartChef::custom_value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);
  check_custom_op_value(operation, "MatrixBandPart");

  /**
   * REGISTER_OP("MatrixBandPart")
    .Input("input: T")
    .Input("num_lower: Tindex")
    .Input("num_upper: Tindex")
    .Output("band: T")
    .Attr("T: type")
    .Attr("Tindex: {int32, int64} = DT_INT64")
    .SetShapeFn(shape_inference::UnchangedShape);
   */

  auto flex_buffers = std::make_unique<flexbuffers::Builder>();
  size_t map_start = flex_buffers->StartMap();

  // TODO Support more data types
  flex_buffers->Int("T", tflite::TensorType_FLOAT32);
  flex_buffers->Int("Tindex", tflite::TensorType_INT64);

  flex_buffers->EndMap(map_start);
  flex_buffers->Finish();

  auto circle_custom_options = fbb.CreateVector(flex_buffers->GetBuffer());
  return circle_custom_options;
}

std::unique_ptr<OpChef> MatrixBandPartChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new MatrixBandPartChef{operation}};
}
