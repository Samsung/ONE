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
 * @file     Builder.h
 * @brief    This file contains Builder structure
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_INTERP_BUILDER_H__
#define __NNFW_TFLITE_INTERP_BUILDER_H__

#include <tensorflow/lite/interpreter.h>

namespace nnfw
{
namespace tflite
{

/**
 * @brief Structure to Builder
 */
struct Builder
{
  /**
   * @brief Destroy the Builder object
   */
  virtual ~Builder() = default;

  /**
   * @brief Build a FlatBuffer model
   * @return The TfLite interpreter object
   */
  virtual std::unique_ptr<::tflite::Interpreter> build(void) const = 0;
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_INTERP_BUILDER_H__
