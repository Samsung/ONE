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

#ifndef __ONERT_IR_TRAIN_INDEX_H__
#define __ONERT_IR_TRAIN_INDEX_H__

#include "ir/Index.h"

#include <cassert>
#include <cstdint>
#include <utility>

namespace onert::ir::train
{

/**
 * @brief Class that provides index of tensor for training
 * @tparam T  Type of index
 */
template <typename T> class TrainingIndex
{
public:
  /**
   * @brief Construct TrainingOperationIndex object.
   * @param index      The operation index
   * @param is_forward Whether the tensor is forward tensor or not
   */
  TrainingIndex() : _index{T{}}, _is_forward{true}
  {
    // DO NOTHING
  }

  /**
   * @brief Construct TrainingOperationIndex object.
   * @tparam T     Type of index
   * @param index      The operation index
   * @param is_forward Whether the tensor is forward tensor or not
   */
  TrainingIndex(const T &index, bool is_forward) : _index{index}, _is_forward{is_forward}
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Get index
   *
   * @return index
   */
  const T &index() const { return _index; }
  /**
   * @brief Get whether the tensor is forward tensor or not
   *
   * @return true if the tensor is forward tensor
   */
  bool is_forward() const { return _is_forward; }

public:
  /**
   * @brief Check if the index is valid or not
   *
   * @return true if the index is valid, false otherwise
   */
  bool valid() const { return _index.valid(); }

public:
  /**
   * @brief operator overloading function for `==`
   *
   * @return Whether two TrainingIndex are equal
   */
  bool operator==(const TrainingIndex &other) const
  {
    return (!_index.valid() && !other.index().valid()) ||
           (_index == other.index() && _is_forward == other.is_forward());
  }
  /**
   * @brief operator overloading function for `!=`
   *
   * @return Whether two TrainingIndex are differenct
   */
  bool operator!=(const TrainingIndex &other) const { return !(*this == other); }

  /**
   * @brief operator overloading function for `<`
   *
   * @return Whether this TrainingIndex is less than other TrainingIndex
   */
  bool operator<(const TrainingIndex &other) const
  {
    return std::hash<TrainingIndex<T>>{}(*this) < std::hash<TrainingIndex<T>>{}(other);
  }

private:
  T _index;
  bool _is_forward;
};

/**
 * @brief Type that provides index of operation node for training
 * @note TrainingOperationIndex can be index of a forwarding node if the member "_is_forward"
 *       of TrainingIndex is true
 *       TrainingOperationIndex can be index of a backwarding node if the membwr "_is_forward"
 *       of TrainingIndex is false
 */
using TrainingOperationIndex = TrainingIndex<OperationIndex>;

/**
 * @brief Type that provides index of operand for training
 * @note TrainingOperandIndex can be index of an operand used in forwarding if the member
 *       "_is_forward" of TrainingIndex is true
 *       TrainingOperandIndex can be index of an operand used in backwarding if the member
 *       "_is_forward" of TrainingIndex is false
 */
using TrainingOperandIndex = TrainingIndex<OperandIndex>;

inline std::ostream &operator<<(std::ostream &o, const TrainingOperationIndex &i)
{
  return operator<<(o, i.index());
}

inline std::ostream &operator<<(std::ostream &o, const TrainingOperandIndex &i)
{
  return operator<<(o, i.index());
}

} // namespace onert::ir::train

namespace std
{

/**
 * @brief Structure that provides hash value of TrainingOperationIndex
 */
template <> struct hash<onert::ir::train::TrainingOperationIndex>
{
  size_t operator()(const onert::ir::train::TrainingOperationIndex &index) const noexcept
  {
    const auto &op_index = index.index();
    const bool is_forward = index.is_forward();

    assert(sizeof(op_index) <= 4);
    assert((op_index.undefined() || op_index.value() < (1 << 16)) &&
           "TrainingOperationIndex's hash creation error, operand_index is too big");
    static_assert(
      sizeof(size_t) >= sizeof(uint32_t),
      "TrainingOperationIndex's hash creation error, size_t size is less than uint32_t");

    return (static_cast<size_t>(op_index.value())) << 16 | static_cast<size_t>(is_forward);
  }
};

} // namespace std

namespace std
{

/**
 * @brief Structure that provides hash value of TrainingOperandIndex
 */
template <> struct hash<onert::ir::train::TrainingOperandIndex>
{
  size_t operator()(const onert::ir::train::TrainingOperandIndex &index) const noexcept
  {
    const auto &operand_index = index.index();
    const bool &is_forward = index.is_forward();

    assert(sizeof(operand_index) <= 4);
    assert((operand_index.undefined() || operand_index.value() < (1 << 16)) &&
           "TrainingOperandIndex's hash creation error, operand_index is too big");
    static_assert(sizeof(size_t) >= sizeof(uint32_t),
                  "TrainingOperandIndex's hash creation error, size_t size is less than uint32_t");

    return (static_cast<size_t>(operand_index.value())) << 16 | static_cast<size_t>(is_forward);
  }
};

} // namespace std

#endif // __ONERT_IR_TRAIN_INDEX_H__
