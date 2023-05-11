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

#ifndef __ONERT_BACKEND_BASIC_TENSOR_BUILDER_H__
#define __ONERT_BACKEND_BASIC_TENSOR_BUILDER_H__

#include <backend/basic/DynamicTensorManager.h>
#include <backend/basic/TensorRegistry.h>
#include <backend/basic/StaticTensorManager.h>

#include <ir/OperandIndexMap.h>

#include "Tensor.h"

#include <unordered_map>

namespace onert
{
namespace backend
{
namespace basic
{

class TensorBuilder
{
public:
  TensorBuilder(const std::shared_ptr<TensorRegistry> &tensor_reg);
  TensorBuilder(const std::shared_ptr<TensorRegistry> &tensor_reg, const std::string planner_id);

  /**
   * @brief     Register tensor information to allocate on CPU backend
   * @param[in] ind    Operand index
   * @param[in] info   Operand information
   * @param[in] layout Operand data layout
   */
  void registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                          ir::Layout backend_layout);

  void notifyFirstUse(const ir::OperandIndex &);
  void notifyLastUse(const ir::OperandIndex &);

  bool isRegistered(const ir::OperandIndex &) const;

  void allocate(void);

  DynamicTensorManager *dynamicTensorManager(void) { return _dynamic_tensor_mgr.get(); }

private:
  const std::shared_ptr<TensorRegistry> _tensor_reg;
  std::unique_ptr<DynamicTensorManager> _dynamic_tensor_mgr;
  std::unique_ptr<StaticTensorManager> _static_tensor_mgr;
  ir::OperandIndexMap<ir::OperandInfo> _tensor_info_map;
};

} // namespace basic
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BASIC_TENSOR_BUILDER_H__
