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
 * @file    IExecutionBuilder.h
 * @ingroup COM_AI_RUNTIME
 * @brief   This file defines interface of ExecutionBuilder
 */
#ifndef __INTERNAL_IEXECUTION_BUILDER_H__
#define __INTERNAL_IEXECUTION_BUILDER_H__

#include <arm_compute/runtime/IFunction.h>

#include <memory>
#include <string>

/**
 * @brief Struct to define interface of ExecutionBuilder
 */
struct IExecutionBuilder
{
  /**
   * @brief Destroy the IExecutionBuilder object
   */
  virtual ~IExecutionBuilder() = default;

  /**
   * @brief     Append function to execute
   * @param[in] name  Name of function
   * @param[in] f     Function to append
   * @return    N/A
   */
  virtual void append(const std::string &name, std::unique_ptr<::arm_compute::IFunction> &&f) = 0;
};

#endif // __INTERNAL_IEXECUTION_BUILDER_H__
