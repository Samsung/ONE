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

#ifndef __ONERT_BACKEND_CONTROLFLOW_TENSOR_BUILDER_H__
#define __ONERT_BACKEND_CONTROLFLOW_TENSOR_BUILDER_H__

#include <backend/cpu_common/StaticTensorManager.h>
#include <backend/cpu_common/TensorRegistry.h>
#include <backend/cpu_common/Tensor.h>

#include <backend/ITensorBuilder.h>
#include <ir/OperandIndexMap.h>

#include <unordered_map>

#include "DynamicTensorManager.h"

namespace onert
{
namespace backend
{
namespace controlflow
{

class TensorBuilder : public ITensorBuilder
{
public:
  TensorBuilder(const std::shared_ptr<TensorRegistry> &tensor_reg);

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

  DynamicTensorManager *dynamicTensorManager(void) override;

  /**
   * @brief Get tensor with a specific OperandIndex.
   * @param ind OperandIndex for the tensor. There must exist a tensor with this ind.
   *        If not, program will crash with assert or exception.
   * @return operand::Tensor *
   */
  cpu_common::Tensor *nativeOwnTensorAt(const ir::OperandIndex &ind);

private:
  const std::shared_ptr<TensorRegistry> _tensor_reg;
  std::unique_ptr<DynamicTensorManager> _dynamic_tensor_mgr;
  std::unique_ptr<cpu_common::StaticTensorManager> _static_tensor_mgr;
  ir::OperandIndexMap<ir::OperandInfo> _tensor_info_map;
  ir::OperandIndexMap<ir::Layout> _tensor_layout_map;
};

} // namespace controlflow
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CONTROLFLOW_TENSOR_BUILDER_H__
