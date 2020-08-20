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

#ifndef __ONERT_IR_OP_SEQUENCES_H__
#define __ONERT_IR_OP_SEQUENCES_H__

#include "ir/Index.h"
#include "ir/OpSequence.h"
#include "util/ObjectManager.h"

namespace onert
{
namespace ir
{

/**
 * @brief Class that manages OpSequence objects
 */
class OpSequences : public util::ObjectManager<OpSequenceIndex, OpSequence>
{
public:
  /**
   * @brief Create an instance of OpSequence with given op and push it to objects
   *
   * @param[in] op_idx Operation index that is emplaced
   * @param[in] layout OpSequence's layout
   * @return OpSequenceIndex
   */
  OpSequenceIndex emplace(const OperationIndex &op_index, Layout layout);

  /**
   * @brief Push an instance of OpSequence to objects
   *
   * @param[in] op_seq An instance of OpSequence
   * @return OpSequenceIndex
   */
  OpSequenceIndex emplace(std::unique_ptr<OpSequence> &&op_seq);
  /**
   * @brief Check if an operation does exist in any OpSequences
   *
   * @param operation_index Operation index to find
   * @return true If such operation exists in any OpSequences otherwise false
   */
  bool containsOperation(const OperationIndex &operation_index) const;
  /**
   * @brief Find an operation from all OpSequences
   *
   * @param operation_index Operation index to find
   * @return OpSequenceIndex Index of OpSequence that contains given operation index
   */
  OpSequenceIndex getOperation(const OperationIndex &operation_index) const;
  /**
   * @brief Remove an operation from OpSequence
   *
   * @param operation_index Operation index to be removed
   */
  void removeFromOpSequence(const OperationIndex &operation_index);

private:
  void cacheSequenceIndex(const OpSequenceIndex &seq_index, const OperationIndex &op_index) const;
  OpSequenceIndex *findSequenceIndex(const OperationIndex &operation_index) const;

  OpSequenceIndex findOperation(const OperationIndex &operation_index) const;
  mutable std::unordered_map<OperationIndex, OpSequenceIndex> _seq_indexes;
};

/**
 * @brief Dump OpSequences
 *
 * @param op_seqs Operation Sequences
 * @param operations Operation context
 */
void dumpOpSequences(const OpSequences &op_seqs, const Operations &operations);

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OP_SEQUENCES_H__
