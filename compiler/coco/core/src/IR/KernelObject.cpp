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

#include "coco/IR/KernelObject.h"
#include "coco/IR/KernelLayouts.h"

#include <nncc/core/ADT/kernel/NCHWLayout.h>

namespace coco
{

KernelObject::KernelObject(const nncc::core::ADT::kernel::Shape &shape)
{
  _layout = KernelLayouts::Generic::create(shape);
}

KernelObject::~KernelObject()
{
  // DO NOTHING
}

const nncc::core::ADT::kernel::Shape &KernelObject::shape(void) const { return _layout->shape(); }

ElemID KernelObject::at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col) const
{
  return _layout->at(n, ch, row, col);
}

} // namespace coco
