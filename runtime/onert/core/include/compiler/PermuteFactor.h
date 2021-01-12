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

/**
 * @file     PermuteFactor.h
 * @brief    This file contains PermuteFactor class
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __ONERT_COMPILER_OPERAND_PERMUTE_FACTOR_H__
#define __ONERT_COMPILER_OPERAND_PERMUTE_FACTOR_H__

#include <functional>

#include "ir/Layout.h"

namespace onert
{
namespace backend
{
class Backend;
} // namespace backend
} // namespace onert

namespace onert
{
namespace compiler
{

/**
 * @brief Class that has factors of permutation
 */
class PermuteFactor
{
public:
  /**
   * @brief Construct PermuteFactor object.
   * @param backend  The backend factor
   * @param layout   The layout factor
   */
  PermuteFactor(const backend::Backend *backend, ir::Layout layout)
    : _backend{backend}, _layout{layout}
  {
    // DO NOTHING
  }
  /**
   * @brief Construct PermuteFactor object by copy semantics.
   */
  PermuteFactor(const PermuteFactor &f) : _backend{f._backend}, _layout{f._layout}
  {
    // DO NOTHING
  }
  /**
   * @brief Construct PermuteFactor object by move semantics.
   */
  PermuteFactor(PermuteFactor &&) = default;

public:
  /**
   * @brief Get backend
   *
   * @return Backend factor
   */
  const backend::Backend *backend() const { return _backend; }
  /**
   * @brief Get layout
   *
   * @return Layout factor
   */
  ir::Layout layout() const { return _layout; }

public:
  /**
   * @brief operator overloading function for `==`
   *
   * @return Whether two PermuteFactor are the same
   */
  bool operator==(const PermuteFactor &other) const
  {
    return _backend == other.backend() && _layout == other.layout();
  }
  /**
   * @brief operator overloading function for `!=`
   *
   * @return Whether two PermuteFactor are differenct
   */
  bool operator!=(const PermuteFactor &other) const { return !(*this == other); }

private:
  const backend::Backend *_backend{nullptr};
  ir::Layout _layout{ir::Layout::UNKNOWN};
};

} // namespace compiler
} // namespace onert

namespace std
{

/**
 * @brief Structure that provides hash value of PermuteFactor
 */
template <> struct hash<onert::compiler::PermuteFactor>
{
  size_t operator()(const onert::compiler::PermuteFactor &factor) const noexcept
  {
    hash<const onert::backend::Backend *> b_hash{};
    hash<onert::ir::Layout> l_hash{};
    return b_hash(factor.backend()) ^ (l_hash(factor.layout()) << 1);
  }
};

} // namespace std

std::ostream &operator<<(std::ostream &os, const onert::compiler::PermuteFactor &obj);

#endif // __ONERT_COMPILER_OPERAND_PERMUTE_FACTOR_H__
