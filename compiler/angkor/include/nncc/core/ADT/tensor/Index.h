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

#ifndef __NNCC_CORE_ADT_TENSOR_INDEX_H__
#define __NNCC_CORE_ADT_TENSOR_INDEX_H__

#include <initializer_list>
#include <vector>
#include <cstdint>

namespace nncc
{
namespace core
{
namespace ADT
{
namespace tensor
{

class Index
{
public:
  Index() = default;
  Index(std::initializer_list<uint32_t> &&l);

public:
  uint32_t rank(void) const;

public:
  Index &resize(uint32_t size);

public:
  Index &fill(uint32_t index);

public:
  uint32_t &at(uint32_t axis);
  uint32_t at(uint32_t axis) const;

private:
  std::vector<uint32_t> _indices;
};

// It throws an exception when rank of inputs does not match.
Index operator+(const Index &lhs, const Index &rhs);
bool operator==(const Index &lhs, const Index &rhs);

} // namespace tensor
} // namespace ADT
} // namespace core
} // namespace nncc

#endif // __NNCC_CORE_ADT_TENSOR_INDEX_H__
