/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SVDF.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> SVDFChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  assert(_operation->has_svdf_options());

  const auto &svdf_options = _operation->svdf_options();

  const auto tflite_activation = as_tflite_activation(svdf_options.activation());

  tflite::SVDFOptionsBuilder svdf_options_builder{fbb};
  svdf_options_builder.add_fused_activation_function(tflite_activation);
  svdf_options_builder.add_asymmetric_quantize_inputs(svdf_options.asymmetric_quantize_inputs());
  svdf_options_builder.add_rank(svdf_options.rank());

  return svdf_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> SVDFChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new SVDFChef{operation}};
}
