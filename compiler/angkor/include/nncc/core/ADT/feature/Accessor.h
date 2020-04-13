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

#ifndef __NNCC_CORE_ADT_FEATURE_ACCESSOR_H__
#define __NNCC_CORE_ADT_FEATURE_ACCESSOR_H__

#include <cstdint>

namespace nncc
{
namespace core
{
namespace ADT
{
namespace feature
{

template <typename T> struct Accessor
{
  virtual ~Accessor() = default;

  virtual T &at(uint32_t ch, uint32_t row, uint32_t col) = 0;
};

} // namespace feature
} // namespace ADT
} // namespace core
} // namespace nncc

#endif // __NNCC_CORE_ADT_FEATURE_ACCESSOR_H__
