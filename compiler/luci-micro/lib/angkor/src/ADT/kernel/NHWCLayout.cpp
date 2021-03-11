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

#include "nncc/core/ADT/kernel/NHWCLayout.h"

using nncc::core::ADT::kernel::Shape;

static uint32_t NHWC_offset(const Shape &shape, uint32_t n, uint32_t ch, uint32_t row, uint32_t col)
{
  return ((n * shape.height() + row) * shape.width() + col) * shape.depth() + ch;
}

namespace nncc
{
namespace core
{
namespace ADT
{
namespace kernel
{

NHWCLayout::NHWCLayout() : Layout{NHWC_offset}
{
  // DO NOTHING
}

} // namespace kernel
} // namespace ADT
} // namespace core
} // namespace nncc
