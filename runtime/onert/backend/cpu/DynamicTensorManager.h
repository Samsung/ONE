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

#ifndef __ONERT_BACKEND_CPU_DYNAMICTENSOR_MANAGER_H__
#define __ONERT_BACKEND_CPU_DYNAMICTENSOR_MANAGER_H__

#include "MemoryManager.h"
#include "TensorRegistry.h"

#include <backend/IDynamicTensorManager.h>
#include <ir/OperandInfo.h>

namespace onert
{
namespace backend
{
namespace cpu
{

/**
 * @brief Class to manage dynamic tensor and its memory
 * @todo  Optimization is needed
 */
class DynamicTensorManager : public backend::IDynamicTensorManager
{
public:
  DynamicTensorManager(const std::shared_ptr<TensorRegistry> &reg);

  virtual ~DynamicTensorManager() = default;

  /**
   * @brief Allocate memory for dynamic tensor.
   *        If allocated memory is already set to the tensor and
   *          if size of existing tensor's memory and new shape is same, memory will not allocated.
   *          if different, previous memory will be deallocated and memory will be allocated.
   */
  void allocate(const ir::OperandIndex &ind, const ir::Shape &new_shape) override;
  void buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &tensor_info);
  void changeShape(const ir::OperandIndex &, const ir::Shape &) override;

private:
  /**
   * @brief Memory manager for dynamic tensor.
   * @todo  DynamicMemoryManager is not optimized. Optimized one is needed
   */
  std::shared_ptr<cpu_common::DynamicMemoryManager> _dynamic_mem_mgr;
  const std::shared_ptr<TensorRegistry> _tensors;
};

} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_DYNAMICTENSOR_MANAGER_H__
