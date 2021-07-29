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
 * @file     NNAPISession.h
 * @brief    This file contains NNAPISession class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_NNAPI_SESSION_H__
#define __NNFW_TFLITE_NNAPI_SESSION_H__

#include "Session.h"

namespace nnfw
{
namespace tflite
{

/**
 * @brief Class to define NNAPI interpreter session which is inherited from Session class
 */
class NNAPISession final : public Session
{
public:
  /**
   * @brief Construct a NNAPISession object with interpreter of TfLite
   * @param[in] interp The TfLite interpreter pointer
   * @note Invoke BuildGraph() of NNAPI delegate from Interpreter
   */
  NNAPISession(::tflite::Interpreter *interp) : _interp{interp}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get TfLite interpreter pointer
   * @return The TfLite interpreter
   */
  ::tflite::Interpreter *interp(void) override { return _interp; }

public:
  /**
   * @brief Prepare the TfLite interpreter session
   * @return @c true if tensor preparation is successful, otherwise @c false
   */
  bool prepare(void) override
  {
    _interp->UseNNAPI(true);

    if (kTfLiteOk != _interp->AllocateTensors())
    {
      return false;
    }

    return true;
  }

  /**
   * @brief Run the Invoke function of NNAPI delegate
   * @return @c true if Invoke() is successful, otherwise @c false
   */
  bool run(void) override { return kTfLiteOk == _interp->Invoke(); }

  /**
   * @brief Tear down TfLite interpreter session
   * @return @c true always
   */
  bool teardown(void) override
  {
    // DO NOTHING
    return true;
  }

private:
  ::tflite::Interpreter *const _interp;
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_NNAPI_SESSION_H__
