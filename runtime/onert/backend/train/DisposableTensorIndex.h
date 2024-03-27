/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_DISPOSABLE_TENSOR_INDEX_H__
#define __ONERT_BACKEND_TRAIN_DISPOSABLE_TENSOR_INDEX_H__

#include <cassert>
#include <functional>

#include "ir/Index.h"

namespace onert
{
namespace backend
{
namespace train
{

/**
 * @brief Class that is index of DisposableTensor
 */
class DisposableTensorIndex
{
public:
  // /**
  //  * @brief Construct a new Index object
  //  */
  // explicit DisposableTensorIndex(void) : _index{ir::OperationIndex::} {}
  // ir::OperationIndex _op_index;
  // ir::OperandIndex _operand_index;
  // /**
  //  * @brief Copy Constructor
  //  *
  //  * @param o Object to be copied
  //  */
  // DisposableTensorIndex(const DisposableTensorIndex &o) = default;

  /**
   * @brief Construct DisposableTensorIndex object.
   * @param op_index      The operation index
   * @param operand_index The operand index
   */
  DisposableTensorIndex(const ir::OperationIndex &op_index, const ir::OperandIndex &operand_index)
    : _op_index{op_index}, _operand_index{operand_index}
  {
    assert(op_index.valid());
    assert(operand_index.valid());
  }

public:
  /**
   * @brief Get Operation index
   *
   * @return Operation index
   */
  const ir::OperationIndex &op_index() const { return _op_index; }
  /**
   * @brief Get operand index
   *
   * @return Operand index
   */
  const ir::OperandIndex &operand_index() const { return _operand_index; }

public:
  /**
   * @brief operator overloading function for `==`
   *
   * @return Whether two DisposableTensorIndex are the same
   */
  bool operator==(const DisposableTensorIndex &other) const
  {
    return _op_index == other.op_index() && _operand_index == other.operand_index();
  }
  /**
   * @brief operator overloading function for `!=`
   *
   * @return Whether two DisposableTensorIndex are differenct
   */
  bool operator!=(const DisposableTensorIndex &other) const { return !(*this == other); }

private:
  ir::OperationIndex _op_index;
  ir::OperandIndex _operand_index;
};

template <typename T> using TempIndexMap = std::unordered_map<DisposableTensorIndex, T>;

inline std::ostream &operator<<(std::ostream &o, const DisposableTensorIndex &i)
{
  return operator<<(o, i.operand_index());
}

} // namespace train
} // namespace backend
} // namespace onert

namespace std
{

/**
 * @brief Structure that provides hash value of DisposableTensorIndex
 */
template <> struct hash<onert::backend::train::DisposableTensorIndex>
{
  size_t operator()(const onert::backend::train::DisposableTensorIndex &index) const noexcept
  {
    const auto op_index = index.op_index();
    const auto operand_index = index.operand_index();

    assert(sizeof(op_index) <= 4);
    assert(sizeof(operand_index) <= 4);
    static_assert(sizeof(size_t) >= sizeof(uint64_t));
    return (static_cast<uint64_t>(op_index.value())) << 32 |
           static_cast<uint64_t>(operand_index.value());
  }
};

} // namespace std

#endif // __ONERT_BACKEND_TRAIN_DISPOSABLE_TENSOR_INDEX_H__
