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
#include <ostream>

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
  PermuteFactor(const backend::Backend *backend) : _backend{backend}
  {
    // DO NOTHING
  }
  /**
   * @brief Construct PermuteFactor object by copy semantics.
   */
  PermuteFactor(const PermuteFactor &f) : _backend{f._backend}
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

public:
  /**
   * @brief operator overloading function for `==`
   *
   * @return Whether two PermuteFactor are the same
   */
  bool operator==(const PermuteFactor &other) const { return _backend == other.backend(); }
  /**
   * @brief operator overloading function for `!=`
   *
   * @return Whether two PermuteFactor are differenct
   */
  bool operator!=(const PermuteFactor &other) const { return !(*this == other); }

private:
  const backend::Backend *_backend{nullptr};
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
    return hash<const onert::backend::Backend *>{}(factor.backend());
  }
};

} // namespace std

std::ostream &operator<<(std::ostream &os, const onert::compiler::PermuteFactor &obj);

#endif // __ONERT_COMPILER_OPERAND_PERMUTE_FACTOR_H__
