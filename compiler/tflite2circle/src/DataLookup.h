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

/**
 * @brief Returns circle builtin_code according to tflite.
 *
 * @note You can see a list of currently supported BuiltinOperator in TFLOperator.lst file.
 */
circle::BuiltinOperator get_circle_builtin_code(tflite::BuiltinOperator tfl_bop);

/**
 * @brief Returns circle TensorType according to tflite.
 *
 * @note You can see a list of currently supported TensorType in TFLTensorType.lst file.
 */
circle::TensorType get_circle_tensortype(tflite::TensorType tfl_tt);

/**
 * @brief Returns circle Padding enum according to tflite.
 */
circle::Padding get_circle_padding(tflite::Padding tfl_p);

/**
 * @brief Returns circle ActivationFunctionType according to tflite.
 *
 * @note You can see a list of currently supported ActivationFunctionType in
 *       TFLActivationFunctionType.lst file.
 */
circle::ActivationFunctionType
get_circle_activation_function_type(tflite::ActivationFunctionType tfl_aft);

/**
 * @brief Returns circle builtin_options according to tflite.
 *
 * @note You can see a list of currently supported BuiltinOptions in
 *       TFLBuiltinOptions.lst file.
 *
 *       This function calls the build_circle_##BuiltinOptions internally(e.g.
 *       build_circle_AbsOptions, build_circle_AddOptions, etc.), so refer to it for a more
 *       detailed implementation.
 */
flatbuffers::Offset<void> get_circle_builtin_options(flatbuffers::FlatBufferBuilder &fb,
                                                     const tflite::Operator *op);

/**
 * @brief Returns circle builtin_options_type according to tflite.
 *
 * @note You can see a list of currently supported BuiltinOptions in TFLBuiltinOptions.lst file.
 */
circle::BuiltinOptions get_circle_builtin_options_type(const tflite::Operator *op);

/**
 * @brief Returns circle MirrorPadMode according to tflite.
 */
circle::MirrorPadMode get_circle_mirrorpad_mode(tflite::MirrorPadMode tfl_mode);

} // namespace tflite2circle

#endif // __DATA_LOOKUP_H__
