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

#include "TransposeConvOptions.h"
#include "DataLookup.h"

#include <cassert>

namespace tflite2circle
{

flatbuffers::Offset<circle::TransposeConvOptions>
build_circle_TransposeConvOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_TransposeConvOptions();
  assert(tflite_builtin_options);
  circle::TransposeConvOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_padding(get_circle_padding(tflite_builtin_options->padding()));
  builtin_options_builder.add_stride_w(tflite_builtin_options->stride_w());
  builtin_options_builder.add_stride_h(tflite_builtin_options->stride_h());
  builtin_options_builder.add_fused_activation_function(
    get_circle_activation_function_type(tflite_builtin_options->fused_activation_function()));
  return builtin_options_builder.Finish();
}

} // namespace tflite2circle
