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

namespace onert::backend::train
{

TensorBuilder::TensorBuilder(const std::shared_ptr<TensorRegistry> &tensor_reg,
                             const exec::train::optimizer::Optimizer *optimizer)
  : _tensor_reg{tensor_reg}, _tensor_mgr{new TensorManager(tensor_reg, optimizer->getVarCount())},
    _optimizer{optimizer}
{
  /* empty */
}

void TensorBuilder::registerTensorInfo(const ir::OperandIndex &index, const ir::OperandInfo &info)
{
  _tensor_info_map.emplace(index, info);
  _as_constants[index] = info.isConstant();

  assert(!info.isDynamic());

  // NOTE For now, whether or not to build operands to trainable tensor depends on whether
  //      the corresponding operand is constant.
  if (_as_constants[index])
  {
    auto tensor = std::make_unique<TrainableTensor>(info);
    _tensor_reg->setTrainableTensor(index, std::move(tensor));
  }
  else
  {
    auto tensor = std::make_unique<Tensor>(info);
    _tensor_reg->setNonConstTensor(index, std::move(tensor));
  }
}

void TensorBuilder::registerBackwardTensorInfo(const ir::OperandIndex &index,
                                               const ir::OperandInfo &info)
{
  _backward_tensor_info_map.emplace(index, info);

  assert(!info.isDynamic());

  // NOTE For now, whether or not to build operands to trainable tensor depends on whether
  //      the corresponding operand is constant.
  assert(_as_constants[index] == info.isConstant());
  if (_as_constants[index])
  {
    auto tensor = std::make_unique<GradientTensor>(info);
    _tensor_reg->setGradientTensor(index, std::move(tensor));

    // Initialize tensors for gradient variables
    for (uint32_t i = 0; i < _optimizer->getVarCount(); ++i)
    {
      auto tensor = std::make_unique<Tensor>(info);
      _tensor_reg->getTrainableTensor(index)->appendOptVar(std::move(tensor));
    }
  }
  else
  {
    auto tensor = std::make_unique<BackPropTensor>(info);
    _tensor_reg->setBackPropTensor(index, std::move(tensor));
  }
}

void TensorBuilder::registerDisposableBackwardTensorInfo(const DisposableTensorIndex &index,
                                                         const ir::OperandInfo &info)
{
  assert(!info.isDynamic());
  assert(!_as_constants[index.operand_index()]);

  auto disposable_tensor = std::make_unique<BackPropTensor>(info);
  _tensor_reg->setDisposableBackPropTensor(index, std::move(disposable_tensor));

  _disposable_backprops.add(index);
}

void TensorBuilder::registerLayerScopeTensor(const LayerScopeTensorIndex &index,
                                             std::shared_ptr<LayerScopeTensor> &tensor)
{
  const auto op_idx = index.op_index();

  const auto pair = _operation_to_layerscope.find(op_idx);
  if (pair == _operation_to_layerscope.end())
  {
    util::Set<LayerScopeTensorIndex> tensor_indices;
    tensor_indices.add(index);
    _operation_to_layerscope[op_idx] = tensor_indices;
  }
  else
  {
    assert(!pair->second.contains(index));
    pair->second.add(index);
  }

  _tensor_reg->setLayerScopeTensor(index, tensor);
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

void TensorBuilder::notifyLastUse(const ir::OperandIndex &index)
{
  if (_as_constants[index])
  {
    _tensor_mgr->releaseTrainablePlan(index);
  }
  else
  {
    _tensor_mgr->releaseNonConstPlan(index);
  }
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

void TensorBuilder::notifyBackwardLastUse(const ir::OperandIndex &index)
{
  if (_as_constants[index])
  {
    _tensor_mgr->releaseGradientPlan(index);
  }
  else
  {
    _tensor_mgr->releaseBackPropPlan(index);
  }
}

void TensorBuilder::notifyDisposableBackPropFirstUse(const DisposableTensorIndex &index)
{
  _tensor_mgr->claimDisposableBackPropPlan(index);
}

void TensorBuilder::notifyDisposableBackPropLastUse(const DisposableTensorIndex &index)
{
  _tensor_mgr->releaseDisposableBackPropPlan(index);
}

void TensorBuilder::notifyLayerScopeFirstUse(const LayerScopeTensorIndex &index)
{
  _tensor_mgr->claimLayerScopePlan(index);
}

void TensorBuilder::notifyLayerScopeLastUse(const LayerScopeTensorIndex &index)
{
  _tensor_mgr->releaseLayerScopePlan(index);
}

bool TensorBuilder::isRegistered(const ir::OperandIndex &index) const
{
  return _tensor_info_map.find(index) != _tensor_info_map.end();
}

bool TensorBuilder::isRegisteredBackward(const ir::OperandIndex &index) const
{
  return _backward_tensor_info_map.find(index) != _backward_tensor_info_map.end();
}

bool TensorBuilder::isRegisteredDisposableBackwardTensor(const DisposableTensorIndex &index) const
{
  return _disposable_backprops.contains(index);
}

bool TensorBuilder::isRegisteredLayerScopeTensor(const ir::OperationIndex &index) const
{
  const auto pair = _operation_to_layerscope.find(index);
  return (pair != _operation_to_layerscope.end());
}

const util::Set<LayerScopeTensorIndex> &
TensorBuilder::getRegisteredLayerScopeTensorIndices(const ir::OperationIndex &index) const
{
  const auto pair = _operation_to_layerscope.find(index);
  assert(pair != _operation_to_layerscope.end());

  return pair->second;
}

LayerScopeTensorLifeTime
TensorBuilder::getLayerScopeTensorLifeTime(const LayerScopeTensorIndex &index) const
{
  const auto &ls_tensors = _tensor_reg->layerscope_tensors();
  const auto &tensor = ls_tensors.at(index);
  return tensor->lifetime();
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
  _tensor_mgr->allocateDisposableBackPropTensors();
}

void TensorBuilder::allocateLayerScope(void) { _tensor_mgr->allocateLayerScopeTensors(); }

} // namespace onert::backend::train
