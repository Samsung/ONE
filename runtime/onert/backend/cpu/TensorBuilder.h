/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CPU_TENSOR_BUILDER_H__
#define __ONERT_BACKEND_CPU_TENSOR_BUILDER_H__

#include <backend/cpu_common/DynamicTensorManager.h>
#include <backend/cpu_common/TensorRegistry.h>

#include <backend/ITensorBuilder.h>
#include <ir/OperandIndexMap.h>

#include "StaticTensorManager.h"
#include "Tensor.h"

#include <unordered_map>

namespace onert
{
namespace backend
{
namespace cpu
{

class TensorBuilder : public ITensorBuilder
{
public:
  TensorBuilder();

  /**
   * @brief     Register tensor information to allocate on CPU backend
   * @param[in] ind    Operand index
   * @param[in] info   Operand information
   * @param[in] layout Operand data layout
   */
  void registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                          ir::Layout backend_layout) override;

  void notifyFirstUse(const ir::OperandIndex &) override;
  void notifyLastUse(const ir::OperandIndex &) override;

  bool isRegistered(const ir::OperandIndex &) const override;

  void prepare(void) override;
  void allocate() override;
  void postFunctionPrepare() override { /* DO NOTHING */}

  std::unique_ptr<ITensorManager> releaseStaticTensorManager(void) override;

  IDynamicTensorManager *dynamicTensorManager(void) override { return _dynamic_tensor_mgr.get(); }

  std::unique_ptr<ITensorManager> releaseDynamicTensorManager(void) override;

  bool setMigrantTensor(const ir::OperandIndex &ind,
                        const std::shared_ptr<IPortableTensor> &tensor) override;

  std::shared_ptr<ITensorRegistry> tensorRegistry() override { return _tensor_reg; }

private:
  const std::shared_ptr<cpu_common::TensorRegistry> _tensor_reg;
  std::unique_ptr<cpu_common::DynamicTensorManager> _dynamic_tensor_mgr;
  std::unique_ptr<StaticTensorManager> _static_tensor_mgr;
  ir::OperandIndexMap<ir::OperandInfo> _tensor_info_map;
};

} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_TENSOR_BUILDER_H__
