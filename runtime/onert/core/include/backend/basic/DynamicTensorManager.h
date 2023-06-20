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

#ifndef __ONERT_BACKEND_BASIC_DYNAMICTENSOR_MANAGER_H__
#define __ONERT_BACKEND_BASIC_DYNAMICTENSOR_MANAGER_H__

#include "MemoryManager.h"
#include "TensorRegistry.h"

#include <ir/OperandInfo.h>
#include <ir/IOperation.h>
#include <ir/Index.h>

#include <unordered_set>

namespace onert
{
namespace backend
{
namespace basic
{

// TODO Find optimized algorithm to manage memory.

/**
 * @brief Class to manage dynamic tensor and its memory
 */
class DynamicTensorManager
{
public:
  DynamicTensorManager(const std::shared_ptr<TensorRegistry> &reg);

  virtual ~DynamicTensorManager() = default;

  void buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &tensor_info,
                   ir::Layout backend_layout);

  std::shared_ptr<DynamicMemoryManager> dynamic_mem_mgr() { return _dynamic_mem_mgr; }

private:
  const ITensor *getRawITensor(ir::OperandIndex ind);

private:
  /**
   * @brief Memory manager for dynamic tensor.
   * @todo  DynamicMemoryManager is not optimized. Optimized one is needed
   */
  std::shared_ptr<DynamicMemoryManager> _dynamic_mem_mgr;
  const std::shared_ptr<TensorRegistry> _tensors;

  // contains list of dynamic tensor index, which can be deallocated after running operation
  // note: this map could contain static tensor index too. Careful use is required.
  std::unordered_map<ir::OperationIndex, std::unordered_set<backend::ITensor *>>
    _dealloc_tensor_map;
};

} // namespace basic
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BASIC_DYNAMICTENSOR_MANAGER_H__
