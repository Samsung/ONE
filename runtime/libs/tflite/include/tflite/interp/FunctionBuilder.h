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
 * @file     FunctionBuilder.h
 * @brief    This file contains FunctionBuilder class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_INTERP_FUNCTION_BUILDER_H__
#define __NNFW_TFLITE_INTERP_FUNCTION_BUILDER_H__

#include <tensorflow/lite/model.h>

#include "tflite/interp/Builder.h"

namespace nnfw
{
namespace tflite
{

/**
 * @brief Class to define FunctionBuilder which is inherited from Builder
 */
class FunctionBuilder final : public Builder
{
public:
  using SetupFunc = std::function<void(::tflite::Interpreter &)>;

public:
  /**
   * @brief Construct a FunctionBuilder object with SetupFunction
   * @param[in] fn The SetupFunc object
   */
  FunctionBuilder(const SetupFunc &fn) : _fn{fn}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Build a SetupFunc
   * @return The TfLite interpreter pointer address
   */
  std::unique_ptr<::tflite::Interpreter> build(void) const override;

private:
  SetupFunc _fn;
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_INTERP_FUNCTION_BUILDER_H__
