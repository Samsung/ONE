/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "util/ShapeInference.h"

namespace onert
{
namespace shape_inference
{

void StaticInferer::visit(const ir::operation::If &op)
{
  auto &then_graph = _lowered_subgs.at(op.param().then_subg_index)->graph();
  auto &else_graph = _lowered_subgs.at(op.param().else_subg_index)->graph();
  const std::vector<ir::OperandIndex> inputs{op.getInputs().begin() + 1, op.getInputs().end()};
  const auto &outputs = op.getOutputs();

  // re-sizing input shapes of then subgraph
  const auto &then_inputs = then_graph.getInputs();
  assert(inputs.size() == then_inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    auto &then_input = then_graph.operands().at(then_inputs.at(i));
    if (_operands.at(inputs.at(i)).info().isDynamic())
    {
      then_input.info().setDynamic();
    }
    else
    {
      auto new_shape = _operands.at(inputs.at(i)).info().shape();
      then_input.info().shape(new_shape);
    }
  }

  // re-sizing input shapes of else subgraph
  const auto &else_inputs = else_graph.getInputs();
  assert(inputs.size() == else_inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    auto &else_input = else_graph.operands().at(else_inputs.at(i));
    if (_operands.at(inputs.at(i)).info().isDynamic())
    {
      else_input.info().setDynamic();
    }
    else
    {
      const auto &new_shape = _operands.at(inputs.at(i)).info().shape();
      else_input.info().shape(new_shape);
    }
  }

  // re-sizing operands of then subgraph
  StaticInferer then_inferer(op.param().then_subg_index, _lowered_subgs);
  _lowered_subgs.at(op.param().then_subg_index)
      ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, const ir::OpSequence &op_seq) {
        then_inferer.infer(op_seq);
      });

  // re-sizing operands of else subgraph
  StaticInferer else_inferer(op.param().else_subg_index, _lowered_subgs);
  _lowered_subgs.at(op.param().else_subg_index)
      ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, const ir::OpSequence &op_seq) {
        else_inferer.infer(op_seq);
      });

  // re-sizing output shapes
  const auto &then_outputs = _lowered_subgs.at(op.param().then_subg_index)->graph().getOutputs();
  const auto &else_outputs = _lowered_subgs.at(op.param().else_subg_index)->graph().getOutputs();
  assert(outputs.size() == then_outputs.size());
  assert(outputs.size() == else_outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i)
  {
    const auto &then_output = then_graph.operands().at(then_outputs.at(i));
    const auto &else_output = else_graph.operands().at(else_outputs.at(i));
    auto &output = _operands.at(outputs.at(i));
    if (!then_output.info().isDynamic() && !else_output.info().isDynamic() &&
        then_output.shape() == else_output.shape())
    {
      output.info().shape(then_output.shape());
    }
    else
    {
      output.info().setDynamic();
    }
  }
}

} // namespace shape_inference
} // namespace onert
