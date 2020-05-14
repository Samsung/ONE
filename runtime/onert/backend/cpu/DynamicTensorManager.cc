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

#include "DynamicTensorManager.h"

namespace onert
{
namespace backend
{
namespace cpu
{

DynamicTensorManager::DynamicTensorManager(const std::shared_ptr<TensorRegistry> &reg)
    : _tensors{reg}
{
  // DO NOTHING
}

void DynamicTensorManager::buildTensor(const ir::OperandIndex &ind,
                                       const ir::OperandInfo &tensor_info)
{
  assert(_tensors->find(ind) == _tensors->end());
  auto tensor = std::make_shared<operand::Tensor>(tensor_info);
  (*_tensors)[ind] = tensor;
}

} // namespace cpu
} // namespace backend
} // namespace onert
