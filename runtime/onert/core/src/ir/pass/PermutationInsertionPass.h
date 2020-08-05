/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_GRAPH_PASS_PERMUTATION_INSERTION_PASS_H__
#define __ONERT_GRAPH_PASS_PERMUTATION_INSERTION_PASS_H__

#include "LoweredOperandPass.h"
#include "compiler/BackendManager.h"
#include "ir/Operand.h" //for OperationIndex
#include "ir/operand/PermuteFactor.h"

namespace onert
{
namespace ir
{
namespace pass
{

class PermutationInsertionPass : public LoweredOperandPass
{
public:
  using LoweredOperandPass::LoweredOperandPass;

public:
  std::string id() override { return "PermutationInsertionPass"; }
  void callback(const OperandIndex &index, Operand &object) override;

private:
  /**
   * @brief Insert Permute operation that has given operand as input
   *
   * @param operand_index is the target operand index for the insertion
   * @param factor is the output operand's backend type and layout
   *
   * @return OperationIndex
   */
  OperationIndex insertPermute(const OperandIndex &operand_index,
                               const operand::PermuteFactor &factor);
};

} // namespace pass
} // namespace ir
} // namespace onert

#endif // __ONERT_GRAPH_PASS_PERMUTATION_INSERTION_PASS_H__
