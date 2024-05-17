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
 * @file     InterpreterSession.h
 * @brief    This file contains InterpreterSession class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_INTERPRETER_SESSION_H__
#define __NNFW_TFLITE_INTERPRETER_SESSION_H__

#include "Session.h"

namespace nnfw
{
namespace tflite
{

/**
 * @brief Class to define TfLite interpreter session which is inherited from Session class
 */
class InterpreterSession final : public Session
{
public:
  /**
   * @brief Construct a InterpreterSession object with interpreter of TfLite
   * @param[in] interp The TfLite interpreter pointer
   */
  InterpreterSession(TfLiteInterpreter *interp) : _interp{interp}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get TfLite interpreter pointer
   * @return The TfLite interpreter
   */
  TfLiteInterpreter *interp(void) override { return _interp; }

public:
  /**
   * @brief Prepare the TfLite interpreter session
   * @return @c true if tensor preparation is successful, otherwise @c false
   */
  bool prepare(void) override
  {
    if (kTfLiteOk != TfLiteInterpreterAllocateTensors(_interp))
    {
      return false;
    }

    return true;
  }

  /**
   * @brief Run the Invoke function of TfLite interpreter
   * @return @c true if Invoke() is successful, otherwise @c false
   */
  bool run(void) override
  {
    // Return true if Invoke returns kTfLiteOk
    return kTfLiteOk == TfLiteInterpreterInvoke(_interp);
  }

  /**
   * @brief Tear down TfLite interpreter session
   * @return @c true always
   */
  bool teardown(void) override
  {
    // Do NOTHING currently
    return true;
  }

private:
  TfLiteInterpreter *const _interp;
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_INTERPRETER_SESSION_H__
