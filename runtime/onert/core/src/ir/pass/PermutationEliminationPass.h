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

#ifndef __ONERT_GRAPH_PASS_PERMUTATION_ELIMINATION_PASS_H__
#define __ONERT_GRAPH_PASS_PERMUTATION_ELIMINATION_PASS_H__

#include "LoweredOperandPass.h"
#include "ir/Operand.h"
#include "ir/OperandIndexSequence.h"

namespace onert
{
namespace ir
{
namespace pass
{

class PermutationEliminationPass : public LoweredOperandPass
{
public:
  using LoweredOperandPass::LoweredOperandPass;

public:
  std::string id() override { return "PermutationEliminationPass"; }

  void callback(const OperandIndex &index, Operand &object) override;

private:
  /**
   * @brief Remove Permute operation that permutates input
   *
   * Note: This function aslo removes model's input and
   * sets output of permutation as model's new input
   *
   * @param inp_index is the target operand index for the elimination
   * @param object is the target operand object for the elimination
   *
   * @return
   */
  void eliminateInput(const OperandIndex &inp_index, Operand &object);

  /**
   * @brief Remove Permute operation that permutates output of a model
   *
   * Note: This function aslo removes model's output and
   * sets input of permutation as model's new output
   *
   * @param out_index is the target operand index for the elimination
   * @param object is the target operand object for the elimination
   *
   * @return
   */
  void eliminateOutput(const OperandIndex &out_index, Operand &object);

  /**
   * @brief Determine if passed operands are permute layer's input and output, that must be
   * eliminated
   *
   * @param inp_index indexes of the input operand to operation
   * @param out_index indexes of the output operand to operation
   * @param is_for_model_input checking for model's input or output
   *
   * @return if it is permutation layer
   */
  bool isPermuteLayerToEliminate(const OperandIndexSequence &inp_indexes,
                                 const OperandIndexSequence &out_indexes, bool is_for_model_input);
};

} // namespace pass
} // namespace ir
} // namespace onert

#endif // __ONERT_GRAPH_PASS_PERMUTATION_ELIMINATION_PASS_H__
