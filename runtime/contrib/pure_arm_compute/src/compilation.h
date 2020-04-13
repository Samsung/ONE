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
 * @file compilation.h
 * @brief This file defines ANeuralNetworksCompilation class for handling Compilation NNAPI
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __COMPILATION_H__
#define __COMPILATION_H__

#include "internal/Model.h"
#include "internal/arm_compute.h"

/**
 * @brief struct to define Compilation of NNAPI
 */
struct ANeuralNetworksCompilation
{
public:
  /**
   * @brief Construct with params
   * @param [in] model Pointer of internal::tflite::Model to set internal::arm_compute::Plan
   */
  ANeuralNetworksCompilation(const std::shared_ptr<const internal::tflite::Model> &model)
      : _plan{new internal::arm_compute::Plan{model}}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get reference of internal::arm_compute::Plan
   * @return Reference of internal::arm_compute::Plan
   */
  internal::arm_compute::Plan &plan(void) { return *_plan; }

public:
  /**
   * @brief Publish internal Plan to param
   * @param [out] plan Pointer of internal::arm_compute::Plan to be set
   * @return N/A
   */
  void publish(std::shared_ptr<const internal::arm_compute::Plan> &plan) { plan = _plan; }
  /**
   * @brief Get @c true if ANeuralNetworksCompilation_finish has been called, otherwise @c false
   * @return @c true if ANeuralNetworksCompilation_finish has been called, otherwise @c false
   */
  bool isFinished(void) { return _isFinished; }
  /**
   * @brief Mark compilation process finished
   * @return N/A
   */
  void markAsFinished() { _isFinished = true; }

private:
  std::shared_ptr<internal::arm_compute::Plan> _plan;
  bool _isFinished{false};
};

#endif
