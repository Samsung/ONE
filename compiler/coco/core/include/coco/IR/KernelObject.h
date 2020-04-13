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

#ifndef __COCO_IR_KERNEL_OBJECT_H__
#define __COCO_IR_KERNEL_OBJECT_H__

#include "coco/IR/Object.h"
#include "coco/IR/KernelLayout.h"
#include "coco/IR/ElemID.h"

#include <nncc/core/ADT/kernel/Shape.h>
#include <nncc/core/ADT/kernel/Layout.h>

namespace coco
{

/**
 * @brief Convolution Kernel (in CNN) values
 */
class KernelObject final : public Object
{
public:
  KernelObject() = default;
  explicit KernelObject(const nncc::core::ADT::kernel::Shape &shape);

public:
  virtual ~KernelObject();

public:
  Object::Kind kind(void) const override { return Object::Kind::Kernel; }

public:
  KernelObject *asKernel(void) override { return this; }
  const KernelObject *asKernel(void) const override { return this; }

public:
  const nncc::core::ADT::kernel::Shape &shape(void) const;

public:
  ElemID at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col) const;

public:
  const KernelLayout *layout(void) const { return _layout.get(); }
  void layout(std::unique_ptr<KernelLayout> &&l) { _layout = std::move(l); }

private:
  std::unique_ptr<KernelLayout> _layout;
};

} // namespace coco

#endif // __COCO_IR_KERNEL_OBJECT_H__
