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

void StaticInferer::visit(const ir::operation::While &op)
{
  auto &cond_graph = _lowered_subgs.at(op.param().cond_subg_index)->graph();
  auto &body_graph = _lowered_subgs.at(op.param().body_subg_index)->graph();
  const auto inputs = op.getInputs();
  const auto &outputs = op.getOutputs();

  // re-sizing input shapes of then subgraph
  const auto &cond_inputs = cond_graph.getInputs();
  assert(inputs.size() == cond_inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    const auto &input = _operands.at(inputs.at(i));
    auto &cond_input = cond_graph.operands().at(cond_inputs.at(i));
    if (input.info().isDynamic())
    {
      cond_input.info().setDynamic();
    }
    else
    {
      auto new_shape = input.info().shape();
      cond_input.info().shape(new_shape);
    }
  }

  // re-sizing input shapes of body subgraph
  const auto &body_inputs = body_graph.getInputs();
  assert(cond_inputs.size() == body_inputs.size());
  for (size_t i = 0; i < cond_inputs.size(); ++i)
  {
    const auto &cond_input = cond_graph.operands().at(cond_inputs.at(i));
    auto &body_input = body_graph.operands().at(body_inputs.at(i));
    if (cond_input.info().isDynamic())
    {
      body_input.info().setDynamic();
    }
    else
    {
      const auto &new_shape = cond_input.info().shape();
      body_input.info().shape(new_shape);
    }
  }

  // re-sizing operands of body subgraph
  StaticInferer body_inferer(op.param().body_subg_index, _lowered_subgs);
  _lowered_subgs.at(op.param().body_subg_index)
      ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, const ir::OpSequence &op_seq) {
        body_inferer.infer(op_seq);
      });

  // Check whether while operation's shapes are predictable
  // If any of shape of body outputs and cond inputs are different, non-constant operands would be
  // set to dynamic
  bool check_unpredictable_dynamic = false;
  const auto &body_outputs = body_graph.getOutputs();
  assert(body_outputs.size() == cond_inputs.size());
  for (size_t i = 0; i < body_outputs.size(); ++i)
  {
    const auto &body_output = body_graph.operands().at(body_outputs.at(i));
    auto &cond_input = cond_graph.operands().at(cond_inputs.at(i));
    if ((cond_input.info().isDynamic() != body_output.info().isDynamic()) ||
        (cond_input.shape() != body_output.shape()))
    {
      check_unpredictable_dynamic = true;
      break;
    }
  }

  if (check_unpredictable_dynamic)
  {
    // Set inputs of body subgraph
    for (const auto &input_index : body_inputs)
    {
      auto &input = body_graph.operands().at(input_index);
      if (!input.isConstant())
      {
        input.info().setDynamic();
      }
    }

    // Set inputs of cond subgraph
    for (const auto &input_index : cond_inputs)
    {
      auto &input = cond_graph.operands().at(input_index);
      if (!input.isConstant())
      {
        input.info().setDynamic();
      }
    }

    // Set non-constant operands of body subgraph to dynamic
    StaticInferer body_inferer(op.param().body_subg_index, _lowered_subgs);
    _lowered_subgs.at(op.param().body_subg_index)
        ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, const ir::OpSequence &op_seq) {
          body_inferer.infer(op_seq);
        });
  }

  // re-sizing operands of cond subgraph
  // If check_unpredictable_dynamic is true, non-constant operands of cond subgraph would be set to
  // dynamic
  StaticInferer cond_inferer(op.param().cond_subg_index, _lowered_subgs);
  _lowered_subgs.at(op.param().cond_subg_index)
      ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, const ir::OpSequence &op_seq) {
        cond_inferer.infer(op_seq);
      });

  // re-sizing outputs of while operation
  // If check_unpredictable_dynamic is true, outputs of while operation would be set to dynamic
  assert(cond_inputs.size() == outputs.size());
  for (size_t i = 0; i < cond_inputs.size(); ++i)
  {
    const auto &cond_input = cond_graph.operands().at(cond_inputs.at(i));
    auto &output = _operands.at(outputs.at(i));
    if (cond_input.info().isDynamic())
    {
      output.info().setDynamic();
    }
    else
    {
      const auto new_shape = cond_input.info().shape();
      output.info().shape(new_shape);
    }
  }
}

} // namespace shape_inference
} // namespace onert
