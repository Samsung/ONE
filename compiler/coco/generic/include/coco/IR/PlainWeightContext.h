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

#ifndef __COCO_IR_PLAIN_WEIGHT_CONTEXT_H__
#define __COCO_IR_PLAIN_WEIGHT_CONTEXT_H__

#include "coco/IR/Bag.h"
#include "coco/IR/KernelObject.h"

#include "coco/ADT/Span.h"

#include <nncc/core/ADT/kernel/Accessor.h>
#include <nncc/core/ADT/kernel/Reader.h>

#include <memory>

namespace coco
{

/**
 * @brief Non-quantized (plain) Weight Data Accessor
 */
template <typename T> struct PlainWeightContext
{
  virtual ~PlainWeightContext() = default;

  /**
   * @brief Allocate a weight space for a given blob
   *
   * @require the following code SHOULD work for any bag "b":
   *   PlainWeightContext<T> ctx;
   *
   *   auto span = ctx.allocate(b);
   *   assert(span.data() != nullptr);
   *   assert(span.size() == bag->size());
   */
  virtual Span<T> allocate(const Bag *) = 0;

  /**
   * @brief Return a pointer to the underlying storage
   *
   * @note weight returns a null-span S for an invalid bag
   *       i.e S.data() == nullptr and S.size() == 0
   */
  virtual Span<T> weight(const Bag *) = 0;

  virtual std::unique_ptr<nncc::core::ADT::kernel::Accessor<T>> access(const KernelObject *) = 0;
  virtual std::unique_ptr<nncc::core::ADT::kernel::Reader<T>> read(const KernelObject *) const = 0;
};

} // namespace coco

#endif // __COCO_IR_PLAIN_WEIGHT_CONTEXT_H__
