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
 * @file     Session.h
 * @brief    This file contains Session class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_SESSION_H__
#define __NNFW_TFLITE_SESSION_H__

#include <tensorflow/lite/c/c_api.h>

namespace nnfw
{
namespace tflite
{

/**
 * @brief Structure to provide interface methods of interpreter session
 */
struct Session
{
  /**
   * @brief  Destruct Session object using default destructor
   */
  virtual ~Session() = default;

  /**
   * @brief   Get the Interpreter object pointer
   * @return  The Interpreter object pointer
   */
  virtual TfLiteInterpreter *interp(void) = 0;

  /**
   * @brief   Prepare the session
   * @return  @c true if prepare method succeeded, otherwise @c false
   */
  virtual bool prepare(void) = 0;
  /**
   * @brief   Run the session
   * @return  @c true if run method succeeded, otherwise @c false
   */
  virtual bool run(void) = 0;
  /**
   * @brief   Teardown(release) the session
   * @return  @c true if teardown method succeeded, otherwise @c false
   */
  virtual bool teardown(void) = 0;
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_INTERP_SESSION_H__
