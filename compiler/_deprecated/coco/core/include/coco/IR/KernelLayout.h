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

#ifndef __COCO_IR_KERNEL_LAYOUT_H__
#define __COCO_IR_KERNEL_LAYOUT_H__

#include "coco/IR/ElemID.h"

#include <nncc/core/ADT/kernel/Shape.h>

namespace coco
{

/**
 * @brief A KernelLayout connectes each kernel index to an element (in a bag)
 *
 * NOTE KernelLayout is an immutable interface
 */
struct KernelLayout
{
  struct ID
  {
    virtual ~ID() = default;
  };

  virtual ~KernelLayout() = default;

  /**
   * @brief Return the identifier of each layout
   *
   * REQUIRED
   *
   * Given l1 and l2 of KernelLayout * type,
   * typeid(*l1) == typeif(*l2) SHOULD hold if l1->id() == l2->id() holds.
   */
  virtual const ID *id(void) const = 0;

  virtual const nncc::core::ADT::kernel::Shape &shape(void) const = 0;

  virtual ElemID at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col) const = 0;
};

} // namespace coco

#endif // __COCO_IR_KERNEL_LAYOUT_H__
