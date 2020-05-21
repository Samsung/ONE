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

/**
 * @file  Convert.h
 * @brief This header declares various as_tflite_TYPE functions
 */
#ifndef __CONVERT_H__
#define __CONVERT_H__

#include <tflchef.pb.h>
#include <mio/tflite/schema_generated.h>

tflite::Padding as_tflite_padding(const tflchef::Padding &value);
tflite::ActivationFunctionType as_tflite_activation(const tflchef::Activation &value);
tflite::TensorType as_tflite_tensortype(const tflchef::TensorType &value);
tflite::MirrorPadMode as_tflite_mirrorpadmode(const tflchef::MirrorPadMode &value);

#endif // __CONVERT_H__
