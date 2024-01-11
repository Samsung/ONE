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

#include "TensorBuilder.h"

#include "Tensor.h"

namespace onert
{
namespace backend
{
namespace train
{

TensorBuilder::TensorBuilder(const std::shared_ptr<TensorRegistry> &tensor_reg,
                             const exec::train::optimizer::Optimizer *optimizer,
                             const std::string planner_id)
  : _tensor_reg{tensor_reg}, _tensor_mgr{new TensorManager(tensor_reg, planner_id)},
    _optimizer{optimizer}
{
  /* empty */
}

void TensorBuilder::registerTensorInfo(const ir::OperandIndex &index, const ir::OperandInfo &info,
                                       ir::Layout layout)
{
  _tensor_info_map.emplace(index, info);
  _as_constants[index] = info.isConstant();

  // Train backend supports only one layout as NHWC
  assert(layout == ir::Layout::NHWC);
  assert(!info.isDynamic());

  // NOTE For now, whether or not to build operands to trainable tensor depends on whether
  //      the corresponding operand is constant.
  if (_as_constants[index])
  {
    auto tensor = std::make_unique<TrainableTensor>(info, layout);
    _tensor_reg->setTrainableTensor(index, std::move(tensor));
  }
  else
  {
    auto tensor = std::make_unique<Tensor>(info, layout);
    _tensor_reg->setNonConstTensor(index, std::move(tensor));
  }
}

void TensorBuilder::registerBackwardTensorInfo(const ir::OperandIndex &index,
                                               const ir::OperandInfo &info, ir::Layout layout)
{
  _backward_tensor_info_map.emplace(index, info);

  // Train backend supports only one layout as NHWC
  assert(layout == ir::Layout::NHWC);
  assert(!info.isDynamic());

  // NOTE For now, whether or not to build operands to trainable tensor depends on whether
  //      the corresponding operand is constant.
  assert(_as_constants[index] == info.isConstant());
  if (_as_constants[index])
  {
    auto tensor = std::make_unique<GradientTensor>(info, layout);
    _tensor_reg->setGradientTensor(index, std::move(tensor));

    // Initialize tensors for gradient variables
    for (uint32_t i = 0; i < _optimizer->getVarCount(); ++i)
    {
      // TODO Optimize memory
      auto tensor = std::make_unique<Tensor>(info, layout);
      tensor->setBuffer(std::make_shared<basic::Allocator>(tensor->total_size()));
      _tensor_reg->getTrainableTensor(index)->appendOptVar(std::move(tensor));
    }
  }
  else
  {
    auto tensor = std::make_unique<BackPropTensor>(info, layout);
    _tensor_reg->setBackPropTensor(index, std::move(tensor));
  }
}

void TensorBuilder::notifyFirstUse(const ir::OperandIndex &index)
{
  // TODO Support momory plan
  if (_as_constants[index])
  {
    _tensor_mgr->claimTrainablePlan(index);
  }
  else
  {
    _tensor_mgr->claimNonConstPlan(index);
  }
}

void TensorBuilder::notifyLastUse(const ir::OperandIndex &)
{
  // TODO Support momory plan
}

void TensorBuilder::notifyBackwardFirstUse(const ir::OperandIndex &index)
{
  // TODO Support momory plan
  if (_as_constants[index])
  {
    _tensor_mgr->claimGradientPlan(index);
  }
  else
  {
    _tensor_mgr->claimBackPropPlan(index);
  }
}

bool TensorBuilder::isRegistered(const ir::OperandIndex &index) const
{
  return _tensor_info_map.find(index) != _tensor_info_map.end();
}

bool TensorBuilder::isRegisteredBackward(const ir::OperandIndex &index) const
{
  return _backward_tensor_info_map.find(index) != _backward_tensor_info_map.end();
}

void TensorBuilder::allocate(void)
{
  _tensor_mgr->allocateNonConstTensors();
  _tensor_mgr->allocateTrainableTensors();
}

void TensorBuilder::allocateBackward(void)
{
  _tensor_mgr->allocateBackPropTensors();
  _tensor_mgr->allocateGradientTensors();
}

} // namespace train
} // namespace backend
} // namespace onert
