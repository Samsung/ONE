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

#ifndef __ONERT_BACKEND_CPU_STATICTENSOR_MANAGER_H__
#define __ONERT_BACKEND_CPU_STATICTENSOR_MANAGER_H__

#include "backend/IStaticTensorManager.h"
#include "backend/cpu_common/DynamicTensorManager.h"
#include "backend/cpu_common/MemoryManager.h"
#include "backend/cpu_common/TensorRegistry.h"
#include "backend/ITensorManager.h"
#include "ir/OperandIndexMap.h"
#include "ir/OperandInfo.h"

namespace onert
{
namespace backend
{
namespace cpu
{

class StaticTensorManager : public backend::IStaticTensorManager
{
public:
  StaticTensorManager(const std::shared_ptr<cpu_common::TensorRegistry> &reg,
                      cpu_common::DynamicTensorManager *dynamic_tensor_manager);
  virtual ~StaticTensorManager() = default;

  void allocateNonconsts(void);
  void deallocateNonconsts(void);

  void buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &tensor_info,
                   ir::Layout backend_layout, bool as_const, unsigned int as_reshape);

  void claimPlan(const ir::OperandIndex &ind, uint32_t size);
  void releasePlan(const ir::OperandIndex &ind);

  void iterate(const std::function<void(const ir::OperandIndex &)> &fn);

private:
  std::unique_ptr<cpu_common::MemoryManager> _nonconst_mgr;
  const std::shared_ptr<cpu_common::TensorRegistry> _tensors;
  ir::OperandIndexMap<bool> _as_constants;
  ir::OperandIndexMap<bool> _as_reshape;
  ir::OperandIndexMap<unsigned int> _as_out_reshape;
  cpu_common::DynamicTensorManager *_dynamic_tensor_manager;
};

} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_STATICTENSOR_MANAGER_H__
