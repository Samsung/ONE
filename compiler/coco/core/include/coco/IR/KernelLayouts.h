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

#ifndef __COCO_IR_KERNEL_LAYOUTS_H__
#define __COCO_IR_KERNEL_LAYOUTS_H__

#include "coco/IR/KernelLayout.h"

#include <nncc/core/ADT/kernel/Layout.h>

#include <vector>
#include <memory>

namespace coco
{
namespace KernelLayouts
{

/**
 * @brief NCHW Kernel Layout
 */
class NCHW final : public KernelLayout
{
private:
  NCHW(const nncc::core::ADT::kernel::Shape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  static const KernelLayout::ID *uid(void);
  const KernelLayout::ID *id(void) const override { return uid(); }

  const nncc::core::ADT::kernel::Shape &shape(void) const override { return _shape; }

  ElemID at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col) const override;

private:
  nncc::core::ADT::kernel::Shape _shape;

public:
  static std::unique_ptr<NCHW> create(const nncc::core::ADT::kernel::Shape &shape);
};

/**
 * @brief NHWC Kernel Layout
 */
class NHWC final : public KernelLayout
{
private:
  NHWC(const nncc::core::ADT::kernel::Shape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  static const KernelLayout::ID *uid(void);
  const KernelLayout::ID *id(void) const override { return uid(); }

  const nncc::core::ADT::kernel::Shape &shape(void) const override { return _shape; }

  ElemID at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col) const override;

private:
  nncc::core::ADT::kernel::Shape _shape;

public:
  static std::unique_ptr<NHWC> create(const nncc::core::ADT::kernel::Shape &shape);
};

/**
 * @brief Generic Kernel Layout
 */
class Generic final : public KernelLayout
{
private:
  Generic(const nncc::core::ADT::kernel::Shape &shape);

public:
  static const KernelLayout::ID *uid(void);
  const KernelLayout::ID *id(void) const override { return uid(); }

  const nncc::core::ADT::kernel::Shape &shape(void) const override { return _shape; }

  ElemID &at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col);
  ElemID at(uint32_t n, uint32_t ch, uint32_t row, uint32_t col) const override;

  void reorder(const nncc::core::ADT::kernel::Layout &l);
  template <typename LayoutImpl> void reorder(void) { reorder(LayoutImpl{}); }

private:
  nncc::core::ADT::kernel::Shape _shape;

private:
  std::vector<ElemID> _content;

public:
  static std::unique_ptr<Generic> create(const nncc::core::ADT::kernel::Shape &shape);
};

} // namespace KernelLayouts
} // namespace coco

#endif // __COCO_IR_KERNEL_LAYOUTS_H__
