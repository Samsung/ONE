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

#ifndef __DATA_LOOKUP_H__
#define __DATA_LOOKUP_H__

#include <mio/tflite/schema_generated.h>
#include <mio/circle/schema_generated.h>

namespace tflite2circle
{

circle::BuiltinOperator get_circle_builtin_code(tflite::BuiltinOperator tfl_bop);
circle::TensorType get_circle_tensortype(tflite::TensorType tfl_tt);
circle::Padding get_circle_padding(tflite::Padding tfl_p);
circle::ActivationFunctionType
get_circle_activation_function_type(tflite::ActivationFunctionType tfl_aft);
flatbuffers::Offset<void> get_circle_builtin_options(flatbuffers::FlatBufferBuilder &fb,
                                                     const tflite::Operator *op);
circle::BuiltinOptions get_circle_builtin_options_type(const tflite::Operator *op);

} // namespace tflite2circle

#endif // __DATA_LOOKUP_H__
