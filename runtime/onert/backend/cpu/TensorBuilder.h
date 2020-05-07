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

#include "DynamicTensorManager.h"
#include "StaticTensorManager.h"
#include "operand/Tensor.h"

#include <backend/ITensorBuilder.h>
#include <ir/OperandIndexMap.h>

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

  bool supportDynamicTensor() override { return true; }

  /**
   * @brief     Register tensor information to allocate on CPU backend
   * @param[in] ind    Operand index
   * @param[in] info   Operand information
   * @param[in] layout Operand data layout
   */
  void registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                          ir::Layout backend_layout, bool as_const) override;

  void notifyFirstUse(const ir::OperandIndex &) override;
  void notifyLastUse(const ir::OperandIndex &) override;

  bool isRegistered(const ir::OperandIndex &) const override;

  void prepare(void) override;
  void allocate() override;
  void postFunctionPrepare() override { /* DO NOTHING */}

  std::shared_ptr<ITensor> tensorAt(const ir::OperandIndex &ind) override;

  void iterate(const IterateFunction &fn) override;

  std::unique_ptr<ITensorManager> releaseStaticTensorManager(void) override;
  std::unique_ptr<ITensorManager> releaseDynamicTensorManager(void) override;

  std::shared_ptr<operand::Tensor> at(const ir::OperandIndex &ind);

private:
  std::unique_ptr<StaticTensorManager> _static_tensor_mgr;
  std::unique_ptr<DynamicTensorManager> _dynamic_tensor_mgr;
  ir::OperandIndexMap<ir::OperandInfo> _tensor_info_map;
  ir::OperandIndexSequence _constants;
};

} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_TENSOR_BUILDER_H__
