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

#include "LocalResponseNormalization.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void>
LocalResponseNormalizationChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_local_response_normalization_options());

  auto &lrn_options = operation.local_response_normalization_options();

  auto tflite_radius = lrn_options.radius();
  auto tflite_bias = lrn_options.bias();
  auto tflite_alpha = lrn_options.alpha();
  auto tflite_beta = lrn_options.beta();

  tflite::LocalResponseNormalizationOptionsBuilder lrn_options_builder{fbb};

  lrn_options_builder.add_radius(tflite_radius);
  lrn_options_builder.add_bias(tflite_bias);
  lrn_options_builder.add_alpha(tflite_alpha);
  lrn_options_builder.add_beta(tflite_beta);

  return lrn_options_builder.Finish().Union();
}

std::unique_ptr<OpChef>
LocalResponseNormalizationChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new LocalResponseNormalizationChef{operation}};
}
