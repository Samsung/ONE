/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Builders.h"

#include "kernels/BatchMatMul.h"
#include <luci/Plan/CircleNodeExecutionPlan.h>

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleBatchMatMul(const luci::CircleNode *circle_node,
                                                       KernelBuilderHelper &helper)
{
  const auto *node = dynamic_cast<const luci::CircleBatchMatMul *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");
  assert(node->arity() == 2);

  const Tensor *lhs = helper.getInputTensor(node->x());
  const Tensor *rhs = helper.getInputTensor(node->y());
  Tensor *output = helper.getOutputTensor(node);

  auto lhs_scratchpad =
    std::make_unique<Tensor>(lhs->element_type(), Shape({}), AffineQuantization{}, "");
  lhs_scratchpad->set_observable(false);
  lhs_scratchpad->set_data_buffer(nullptr);
  auto rhs_scratchpad =
    std::make_unique<Tensor>(rhs->element_type(), Shape({}), AffineQuantization{}, "");
  rhs_scratchpad->set_observable(false);
  rhs_scratchpad->set_data_buffer(nullptr);
  // If node has execution plan then read memory offsets for scratchpad temporary tensor
  // from the beginning of shared memory buffer.
  // Used in Static Memory Manager.
  // TODO move tensors offset initialization to one place
  if (luci::has_execution_plan(node))
  {
    const auto execution_plan = luci::get_execution_plan(node);
    // Check whether the offset for the current BatchMatMul temporary was found.
    if (execution_plan.offsets().size() > 1)
    {
      assert(execution_plan.offsets().size() == 3);

      // If this is true, then we keep this offset in scratchpad.
      lhs_scratchpad->set_offset(execution_plan.offsets().at(1));
      rhs_scratchpad->set_offset(execution_plan.offsets().at(2));
    }
  }
  Tensor *lhs_tmp = helper.getRuntimeGraph(node->graph())->addTensor(std::move(lhs_scratchpad));
  Tensor *rhs_tmp = helper.getRuntimeGraph(node->graph())->addTensor(std::move(rhs_scratchpad));

  BatchMatMulParams params;
  params.adj_x = node->adj_x();
  params.adj_y = node->adj_y();

  return std::make_unique<kernels::BatchMatMul>(lhs, rhs, output, lhs_tmp, rhs_tmp, params);
}

} // namespace luci_interpreter
