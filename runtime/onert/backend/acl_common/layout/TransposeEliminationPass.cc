/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TransposeEliminationPass.h"

#include "LayoutUtils.h"

#include <ir/OperationIndexMap.h>
#include <ir/operation/Transpose.h>
#include <misc/polymorphic_downcast.h>
#include <util/Utils.h>

#include <algorithm>

namespace onert
{
namespace backend
{
namespace acl_common
{

namespace
{

bool isLayoutToNCHW(const ir::operation::Transpose &op, const ir::Operands &operands)
{
  const auto &in_operand_index = op.getInputs().at(ir::operation::Transpose::Input::INPUT);
  const auto &out_operand_index = op.getOutputs().at(0);

  const auto &in_operand = operands.at(in_operand_index);
  const auto &out_operand = operands.at(out_operand_index);

  const auto in_layout = in_operand.info().layout();
  const auto out_layout = out_operand.info().layout();

  const auto is_perm_layout_to_nchw =
    in_layout == ir::Layout::NHWC && out_layout == ir::Layout::NCHW;
  assert(!is_perm_layout_to_nchw ||
         (in_operand.shape().rank() == 4 && out_operand.shape().rank() == 4));

  // NOTE This function assumes that permutation data is always {0, 3, 1, 2} if nhwc to nchw,
  //      but we have no way of checking it unless the operand for permutation is constant
  const auto &perm_operand =
    operands.at(op.getInputs().at(ir::operation::Transpose::Input::PERMUTATION));
  if (is_perm_layout_to_nchw && perm_operand.isConstant())
  {
    assert(perm_operand.typeInfo().type() == ir::DataType::INT32);
    const auto perm_vec = perm_operand.asVector<int32_t>();
    UNUSED_RELEASE(perm_vec);
    assert(perm_vec.size() == 4 &&
           std::equal(std::begin(PERMUTATION_DATA_TO_NCHW), std::end(PERMUTATION_DATA_TO_NCHW),
                      std::begin(perm_vec)));
  }

  return is_perm_layout_to_nchw;
}

bool isLayoutToNHWC(const ir::operation::Transpose &op, const ir::Operands &operands)
{
  const auto &in_operand_index = op.getInputs().at(ir::operation::Transpose::Input::INPUT);
  const auto &out_operand_index = op.getOutputs().at(0);

  const auto &in_operand = operands.at(in_operand_index);
  const auto &out_operand = operands.at(out_operand_index);

  const auto in_layout = in_operand.info().layout();
  const auto out_layout = out_operand.info().layout();

  const auto is_perm_layout_to_nhwc =
    in_layout == ir::Layout::NCHW && out_layout == ir::Layout::NHWC;
  assert(!is_perm_layout_to_nhwc ||
         (in_operand.shape().rank() == 4 && out_operand.shape().rank() == 4));

  // NOTE This function assumes that permutation data is always {0, 2, 3, 1} if nchw to nhwc,
  //      but we have no way of checking it unless the operand for permutation is constant
  const auto &perm_operand =
    operands.at(op.getInputs().at(ir::operation::Transpose::Input::PERMUTATION));
  if (is_perm_layout_to_nhwc && perm_operand.isConstant())
  {
    assert(perm_operand.typeInfo().type() == ir::DataType::INT32);
    const auto perm_vec = perm_operand.asVector<int32_t>();
    UNUSED_RELEASE(perm_vec);
    assert(perm_vec.size() == 4 &&
           std::equal(std::begin(PERMUTATION_DATA_TO_NHWC), std::end(PERMUTATION_DATA_TO_NHWC),
                      std::begin(perm_vec)));
  }

  return is_perm_layout_to_nhwc;
}

bool isPermDataToNCHW(const ir::Operand &perm_operand)
{
  if (perm_operand.isConstant())
  {
    assert(perm_operand.typeInfo().type() == ir::DataType::INT32);
    const auto perm_vec = perm_operand.asVector<int32_t>();
    return perm_vec.size() == 4 &&
           std::equal(std::begin(PERMUTATION_DATA_TO_NCHW), std::end(PERMUTATION_DATA_TO_NCHW),
                      std::begin(perm_vec));
  }
  return false;
}

bool isPermDataToNHWC(const ir::Operand &perm_operand)
{
  if (perm_operand.isConstant())
  {
    assert(perm_operand.typeInfo().type() == ir::DataType::INT32);
    const auto perm_vec = perm_operand.asVector<int32_t>();
    return perm_vec.size() == 4 &&
           std::equal(std::begin(PERMUTATION_DATA_TO_NHWC), std::end(PERMUTATION_DATA_TO_NHWC),
                      std::begin(perm_vec));
  }
  return false;
}

bool isToNCHW(const ir::operation::Transpose &op, const ir::Operands &operands)
{
  const auto &perm_operand_index = op.getInputs().at(ir::operation::Transpose::Input::PERMUTATION);
  const auto &perm_operand = operands.at(perm_operand_index);

  return isLayoutToNCHW(op, operands) || isPermDataToNCHW(perm_operand);
}

bool isToNHWC(const ir::operation::Transpose &op, const ir::Operands &operands)
{
  const auto &perm_operand_index = op.getInputs().at(ir::operation::Transpose::Input::PERMUTATION);
  const auto &perm_operand = operands.at(perm_operand_index);

  return isLayoutToNHWC(op, operands) || isPermDataToNHWC(perm_operand);
}

} // namespace

void TransposeEliminationPass::run()
{
  eliminateTwofold4DTransposeOps();

  _graph.verify();
}

void TransposeEliminationPass::eliminateTwofold4DTransposeOps()
{
  auto &operations = _graph.operations();
  const auto &operands = _graph.operands();

  operations.iterate([&](const ir::OperationIndex &, ir::IOperation &op) {
    if (op.opcode() != ir::OpCode::Transpose)
      return;

    auto &upper_op = dynamic_cast<ir::operation::Transpose &>(op);
    const auto &out_operand_index = upper_op.getOutputs().at(0);
    const auto &candidate_ops = operands.at(out_operand_index).getUses();
    for (const auto &candidate_op_index : candidate_ops)
    {
      auto &candidate_op = operations.at(candidate_op_index);
      if (candidate_op.opcode() == ir::OpCode::Transpose)
      {
        auto &lower_op = dynamic_cast<ir::operation::Transpose &>(candidate_op);
        if ((isToNCHW(upper_op, operands) && isToNHWC(lower_op, operands)) ||
            (isToNHWC(upper_op, operands) && isToNCHW(lower_op, operands)))
        {
          foldTransposeOps(upper_op, lower_op);
        }
      }
    }
  });
}

void TransposeEliminationPass::foldTransposeOps(const ir::operation::Transpose &upper_op,
                                                const ir::operation::Transpose &lower_op)
{
  auto &operations = _graph.operations();
  auto &operands = _graph.operands();

  const auto &upper_operand_index = upper_op.getInputs().at(ir::operation::Transpose::Input::INPUT);
  const auto &middle_operand_index = upper_op.getOutputs().at(0);
  assert(middle_operand_index == lower_op.getInputs().at(ir::operation::Transpose::Input::INPUT));
  const auto &lower_operand_index = lower_op.getOutputs().at(0);
  const auto &upper_perm_operand_index =
    upper_op.getInputs().at(ir::operation::Transpose::Input::PERMUTATION);
  const auto &lower_perm_operand_index =
    lower_op.getInputs().at(ir::operation::Transpose::Input::PERMUTATION);

  auto &upper_operand = operands.at(upper_operand_index);
  auto &middle_operand = operands.at(middle_operand_index);
  auto &lower_operand = operands.at(lower_operand_index);
  auto &lower_perm_operand = operands.at(lower_perm_operand_index);

  const auto &upper_op_index = middle_operand.getDef();
  const auto &lower_op_index = lower_operand.getDef();

  // Append uses of lower operand(output of lower op) to uses of upper operand(input of upper op)
  for (const auto &use : lower_operand.getUses())
    upper_operand.insertUse(use);

  // Remove lower op from uses of middle operand(input of lower op) and permtation input
  middle_operand.removeUse(lower_op_index);
  lower_perm_operand.removeUse(lower_op_index);

  // Eliminate permutation operand of lower op if it's constant
  if (lower_perm_operand.isConstant())
  {
    assert(lower_perm_operand.getUses().size() == 0);
    operands.remove(lower_perm_operand_index);
  }

  // Eliminate lower operand
  operands.remove(lower_operand_index);

  // Eliminate lower op
  operations.remove(lower_op_index);

  // Eliminate upper op and inputs/outputs if uses of middle operand is empty
  if (middle_operand.getUses().size() == 0)
  {
    upper_operand.removeUse(upper_op_index);
    operands.remove(middle_operand_index);
    if (lower_perm_operand.isConstant())
    {
      assert(lower_perm_operand.getUses().size() == 0);
      operands.remove(upper_perm_operand_index);
    }
    operations.remove(upper_op_index);
  }
}

} // namespace acl_common
} // namespace backend
} // namespace onert
