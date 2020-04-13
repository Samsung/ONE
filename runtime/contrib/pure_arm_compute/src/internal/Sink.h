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
 * @file    Sink.h
 * @ingroup COM_AI_RUNTIME
 * @brief   This file defines Sink struct
 */
#ifndef __INTERNAL_SINK_H__
#define __INTERNAL_SINK_H__

#include <arm_compute/core/ITensor.h>

/**
 * @brief Struct to get tensor data from arm compute tensor (abstract)
 */
struct Sink
{
  /**
   * @brief Destroy the Sink object
   */
  virtual ~Sink() = default;

  /**
   * @brief     Get tensor data from arm compute tensor
   * @param[in] tensor  Tensor object of arm compute to get data
   * @return    N/A
   */
  virtual void pull(::arm_compute::ITensor &tensor) const = 0;
};

#endif // __INTERNAL_SINK_H__
