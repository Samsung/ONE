/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_MISC_POLYMORPHIC_DOWNCAST_H__
#define __NNFW_MISC_POLYMORPHIC_DOWNCAST_H__

#include <cassert>
#include <memory>

namespace nnfw
{
namespace misc
{

template <typename DstType, typename SrcType> inline DstType polymorphic_downcast(SrcType *x)
{
  assert(dynamic_cast<DstType>(x) == x);
  return static_cast<DstType>(x);
}

template <typename DstType, typename SrcType> inline DstType polymorphic_downcast(SrcType &x)
{
  assert(std::addressof(dynamic_cast<DstType>(x)) == std::addressof(x));
  return static_cast<DstType>(x);
}

} // namespace misc
} // namespace nnfw

#endif // __NNFW_MISC_POLYMORPHIC_DOWNCAST_H__
