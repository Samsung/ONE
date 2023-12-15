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

#include "TransposeInsertionPass.h"

#include "LayoutUtils.h"

#include <ir/operation/Transpose.h>

namespace onert
{
namespace backend
{
namespace acl_common
{

namespace
{

bool isPermutationRequired(const ir::Operand &operand, ir::Layout backend_layout)
{
  const auto &shape = operand.shape();
  const auto layout = operand.info().layout();
  return shape.rank() > 4 && layout != ir::Layout::UNKNOWN &&
         backend_layout != ir::Layout::UNKNOWN && layout != backend_layout;
}

} // namespace

void TransposeInsertionPass::run(ir::Layout backend_layout)
{
  insertTransposeOps(backend_layout);

  _graph.verify();
}

void TransposeInsertionPass::insertTransposeOps(ir::Layout backend_layout)
{
  auto &operands = _graph.operands();

  operands.iterate([&](const ir::OperandIndex &index, ir::Operand &operand) {
    // TODO Constant folding if operand is constant
    if (!isPermutationRequired(operand, backend_layout))
      return;

    const auto &shape = operand.info().shape();
    if (shape.rank() != 4)
      throw std::runtime_error("TransposeInsertionPass: Unsuppported rank for inserting transpose "
                               "to resolve layout issue");

    const auto layout = operand.info().layout();
    if ((layout == ir::Layout::NHWC && backend_layout == ir::Layout::NCHW) ||
        (layout == ir::Layout::NCHW && backend_layout == ir::Layout::NHWC))
      insert4DTransposeOp(index, backend_layout);
    // TODO Replace calling ir::to_string with calling an internal function in acl backends
    else
      throw std::runtime_error(
        std::string("TransposeInsertionPass: Unsupported layout. frontend layout : ") +
        ir::to_string(layout) + std::string(", backend layout : ") + ir::to_string(backend_layout));
  });
}

void TransposeInsertionPass::insert4DTransposeOp(ir::OperandIndex in_operand_index,
                                                 ir::Layout output_layout)
{
  auto &in_operand = _graph.operands().at(in_operand_index);

  // Generate permutation operand
  const auto perm_operand_index =
    _graph.addOperand(ir::Shape{4}, ir::TypeInfo{ir::DataType::INT32});
  _graph.operands().at(perm_operand_index).info().setAsConstant();

  std::shared_ptr<ir::ExternalData> perm_data{};
  if (output_layout == ir::Layout::NCHW)
  {
    assert(in_operand.info().layout() == ir::Layout::NCHW);
    auto perm_data = std::make_shared<ir::ExternalData>(
      reinterpret_cast<const uint8_t *>(PERMUTATION_DATA_TO_NCHW),
      sizeof(PERMUTATION_DATA_TO_NCHW));
  }
  else if (output_layout == ir::Layout::NHWC)
  {
    assert(in_operand.info().layout() == ir::Layout::NHWC);
    auto perm_data = std::make_shared<ir::ExternalData>(
      reinterpret_cast<const uint8_t *>(PERMUTATION_DATA_TO_NHWC),
      sizeof(PERMUTATION_DATA_TO_NHWC));
  }
  else
    assert(false && "TransposeInsertionPass: Unsupported layout for 4D");

  _graph.operands().at(perm_operand_index).data(perm_data);

  // Generate output operand
  const auto out_operand_index = _graph.addOperand(in_operand.shape(), in_operand.typeInfo());
  _graph.operands().at(out_operand_index).info().layout(output_layout);

  // Insert transpose operation to the graph
  ir::OperandIndexSequence inputs{in_operand_index, perm_operand_index};
  ir::OperandIndexSequence outputs{out_operand_index};
  auto transpose_op = std::make_unique<ir::operation::Transpose>(inputs, outputs);
  auto transpose_op_index = _graph.operations().push(std::move(transpose_op));

  VERBOSE_F() << "Transpose Op inserted, node index : " << transpose_op_index << std::endl;
  VERBOSE_F() << "  - Input (original) Operand : " << in_operand_index << "("
              << ir::to_string(in_operand.info().layout()) << ")" << std::endl;
  VERBOSE_F() << "  - Output(inserted) Operand : " << out_operand_index << "("
              << ir::to_string(output_layout) << ")" << std::endl;

  // Update Use/Def info
  {
    auto &in_operand = _graph.operands().at(in_operand_index);
    auto &perm_operand = _graph.operands().at(perm_operand_index);
    auto &out_operand = _graph.operands().at(out_operand_index);

    const auto uses = in_operand.getUses();

    in_operand.clearUses();
    in_operand.insertUse(transpose_op_index);

    perm_operand.insertUse(transpose_op_index);

    for (const auto &i : uses)
      out_operand.insertUse(i);
    out_operand.setDef(transpose_op_index);
  }
}

} // namespace acl_common
} // namespace backend
} // namespace onert
