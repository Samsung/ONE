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

#include "MaxPoolWithArgmax.h"
#include "OpUtils.h"

#include <flatbuffers/flexbuffers.h>

flatbuffers::Offset<void> MaxPoolWithArgmaxChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  return flatbuffers::Offset<void>();
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
MaxPoolWithArgmaxChef::custom_value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);
  check_custom_op_value(operation, "MaxPoolWithArgmax");

  /**
   * REGISTER_OP("MaxPoolWithArgmax")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr("Targmax: {int32, int64} = DT_INT64")
    .Attr(GetPaddingAttrString())
    .Attr("include_batch_in_index: bool = false")
    .Input("input: T")
    .Output("output: T")
    .Output("argmax: Targmax")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolShape(c));
      c->set_output(1, c->output(0));
      return Status::OK();
    });
   */

  auto flex_buffers = std::make_unique<flexbuffers::Builder>();
  size_t map_start = flex_buffers->StartMap();

  auto start = flex_buffers->StartVector("ksize");
  flex_buffers->Add(1);
  flex_buffers->Add(operation.max_pool_with_argmax_options().filter_width());
  flex_buffers->Add(operation.max_pool_with_argmax_options().filter_height());
  flex_buffers->Add(1);
  flex_buffers->EndVector(start, /*typed=*/true, /*fixed=*/false);
  start = flex_buffers->StartVector("strides");
  flex_buffers->Add(1);
  flex_buffers->Add(operation.max_pool_with_argmax_options().stride_w());
  flex_buffers->Add(operation.max_pool_with_argmax_options().stride_h());
  flex_buffers->Add(1);
  flex_buffers->EndVector(start, /*typed=*/true, /*fixed=*/false);
  auto output_type = operation.max_pool_with_argmax_options().output_type();
  assert(output_type == tflchef::INT64 || output_type == tflchef::INT32);
  flex_buffers->Int("Targmax", output_type);
  std::string padding = operation.max_pool_with_argmax_options().padding() ? "VALID" : "SAME";
  flex_buffers->String("padding", padding);
  flex_buffers->Bool("include_batch_in_index",
                     operation.max_pool_with_argmax_options().include_batch_in_index());
  flex_buffers->Int("T", tflchef::FLOAT32);
  flex_buffers->EndMap(map_start);
  flex_buffers->Finish();

  auto circle_custom_options = fbb.CreateVector(flex_buffers->GetBuffer());
  return circle_custom_options;
}

std::unique_ptr<OpChef>
MaxPoolWithArgmaxChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new MaxPoolWithArgmaxChef{operation}};
}
