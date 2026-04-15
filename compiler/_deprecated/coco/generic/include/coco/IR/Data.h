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

#ifndef __COCO_IR_DATA_H__
#define __COCO_IR_DATA_H__

#include "coco/IR/PlainWeightContext.h"

#include <memory>

namespace coco
{

/**
 * @brief Core coco entity for constant weights
 */
struct Data
{
  virtual ~Data() = default;

  /**
   * @brief Return true if a given bag has an allocated weight data
   */
  virtual bool allocated(const coco::Bag *) const = 0;

  /**
   * @brief Release a memory chunk allocated for weight data of a given bag
   *
   * WARN Do NOT invoke release for a bag "b" for which allocated(b) does NOT hold
   */
  virtual void release(const coco::Bag *) = 0;

  virtual PlainWeightContext<float> *f32(void) = 0;
  virtual const PlainWeightContext<float> *f32(void) const = 0;

  static std::unique_ptr<Data> create(void);
};

} // namespace coco

#endif // __COCO_IR_DATA_H__
