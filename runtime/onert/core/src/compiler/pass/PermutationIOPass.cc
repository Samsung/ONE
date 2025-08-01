/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PermutationIOPass.h"

#include "compiler/BackendManager.h"
#include "util/Utils.h"

namespace onert
{
namespace compiler
{
namespace pass
{

void PermutationIOPass::run()
{
  if (_options.input_layout.size() == 0 && _options.output_layout.size() == 0 &&
      _options.input_type.size() == 0 && _options.output_type.size() == 0)
    return;

  for (auto i = ir::IOIndex{0}; i < ir::IOIndex{_graph.getInputs().size()}; i++)
  {
    if (_options.input_layout.count(i) == 0 && _options.input_type.count(i) == 0)
      continue;

    const auto index = _graph.getInputs().at(i);
    const auto &type = _options.input_type.count(i) > 0 ? _options.input_type.at(i)
                                                        : _graph.operands().at(index).typeInfo();
    const auto layout =
      _options.input_layout.count(i) > 0 ? _options.input_layout.at(i) : ir::Layout::NHWC;

    insertInputPermute(index, type, layout);
  }

  for (auto i = ir::IOIndex{0}; i < ir::IOIndex{_graph.getOutputs().size()}; i++)
  {
    if (_options.output_layout.count(i) == 0 && _options.output_type.count(i) == 0)
      continue;

    const auto index = _graph.getOutputs().at(i);
    const auto &type = _options.output_type.count(i) > 0 ? _options.output_type.at(i)
                                                         : _graph.operands().at(index).typeInfo();
    const auto layout =
      _options.output_layout.count(i) > 0 ? _options.output_layout.at(i) : ir::Layout::NHWC;

    insertOutputPermute(index, type, layout);
  }
}

void PermutationIOPass::insertInputPermute(const ir::OperandIndex &index, const ir::TypeInfo &type,
                                           const ir::Layout &from_layout)
{
  assert(from_layout == ir::Layout::NCHW || from_layout == ir::Layout::NHWC);
  const auto &origin_operand = _graph.operands().at(index);

  // Check nothing to permute
  if (origin_operand.typeInfo() == type && from_layout == ir::Layout::NHWC)
    return;

  // Update graph operand
  auto input_shape = from_layout == ir::Layout::NCHW
                       ? ir::convertShape(origin_operand.shape(), ir::PermuteType::NHWC_TO_NCHW)
                       : origin_operand.shape();
  auto input_operand_index = _graph.addOperand(input_shape, type);
  _graph.getInputs().replace(index, input_operand_index);

  // Update graph operation
  using Permute = ir::operation::Permute;
  auto permute_type =
    from_layout == ir::Layout::NCHW ? ir::PermuteType::NCHW_TO_NHWC : ir::PermuteType::SAME;
  auto permute_node = std::make_unique<Permute>(input_operand_index, index, permute_type);
  auto node_index = _graph.operations().push(std::move(permute_node));

  // Update use/def info
  auto &new_input = _graph.operands().at(input_operand_index);
  new_input.setDef(origin_operand.getDef());
  new_input.insertUse(node_index);
  _graph.operands().at(index).setDef(node_index);

  VERBOSE(PermuteIOPass) << "Permute Op inserted for an input, node index : " << node_index
                         << std::endl;
  VERBOSE(PermuteIOPass) << "  - Input (inserted) Operand : " << input_operand_index << std::endl;
  VERBOSE(PermuteIOPass) << "  - Output(original) Operand : " << index << std::endl;
}

void PermutationIOPass::insertOutputPermute(const ir::OperandIndex &index, const ir::TypeInfo &type,
                                            const ir::Layout &to_layout)
{
  assert(to_layout == ir::Layout::NCHW || to_layout == ir::Layout::NHWC);
  auto &origin_operand = _graph.operands().at(index);

  // Check nothing to permute
  if (origin_operand.typeInfo() == type && to_layout == ir::Layout::NHWC)
    return;

  // Update graph operand
  auto output_shape = to_layout == ir::Layout::NCHW
                        ? ir::convertShape(origin_operand.shape(), ir::PermuteType::NHWC_TO_NCHW)
                        : origin_operand.shape();
  auto output_operand_index = _graph.addOperand(output_shape, type);
  _graph.getOutputs().replace(index, output_operand_index);

  // Update graph operation
  using Permute = ir::operation::Permute;
  auto permute_type =
    to_layout == ir::Layout::NCHW ? ir::PermuteType::NHWC_TO_NCHW : ir::PermuteType::SAME;
  auto permute_node = std::make_unique<Permute>(index, output_operand_index, permute_type);
  auto node_index = _graph.operations().push(std::move(permute_node));

  // Update use/def info
  auto &new_output = _graph.operands().at(output_operand_index);
  new_output.setDef(node_index);
  assert(_graph.operands().at(index).getUses().size() == 0);
  origin_operand.insertUse(node_index);

  VERBOSE(PermuteIOPass) << "Permute Op inserted for an output, node index : " << node_index
                         << std::endl;
  VERBOSE(PermuteIOPass) << "  - Input (original) Operand : " << index << std::endl;
  VERBOSE(PermuteIOPass) << "  - Output(inserted) Operand : " << output_operand_index << std::endl;
}

} // namespace pass
} // namespace compiler
} // namespace onert
