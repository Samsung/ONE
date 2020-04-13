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

#ifndef __NNCC_CORE_ADT_KERNEL_INDEX_ENUMERATOR_H__
#define __NNCC_CORE_ADT_KERNEL_INDEX_ENUMERATOR_H__

#include "nncc/core/ADT/kernel/Shape.h"

namespace nncc
{
namespace core
{
namespace ADT
{
namespace kernel
{

class IndexEnumerator
{
public:
  explicit IndexEnumerator(const Shape &shape);

public:
  IndexEnumerator(IndexEnumerator &&) = delete;
  IndexEnumerator(const IndexEnumerator &) = delete;

public:
  bool valid(void) const;

public:
  uint32_t count(void) const;
  uint32_t depth(void) const;
  uint32_t height(void) const;
  uint32_t width(void) const;

public:
  void advance(void);

private:
  // Store max and current offset for count/depth/height/width
  //
  // NOTE Here explicit array is used instead of kernel::Shape to make
  //      a room for improvement such as enumeration order (NHWC, NCHW)
  //      support
  uint32_t _max[4];
  uint32_t _cur[4];

private:
  uint32_t _cursor;
};

} // namespace kernel
} // namespace ADT
} // namespace core
} // namespace nncc

#endif // __NNCC_CORE_ADT_KERNEL_INDEX_ENUMERATOR_H__
