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
 * @file Source.h
 * @brief This file contains Source struct for pushing ITensor
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __INTERNAL_SOURCE_H__
#define __INTERNAL_SOURCE_H__

#include <arm_compute/core/ITensor.h>

/**
 * @brief Struct to push inner source to ITensor.
 */
struct Source
{
  /**
   * @brief Destructor as default
   */
  virtual ~Source() = default;

  /**
   * @brief Push inner source to ITensor
   * @param [in] tensor ITensor to be pushed into
   * @return N/A
   */
  virtual void push(::arm_compute::ITensor &tensor) const = 0;
};

#endif // __INTERNAL_SOURCE_H__
