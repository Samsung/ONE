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

#ifndef __ONERT_BACKEND_TRAIN_TENSOR_BUILDER_H__
#define __ONERT_BACKEND_TRAIN_TENSOR_BUILDER_H__

#include "TensorManager.h"
#include "TensorRegistry.h"
#include "DisposableTensorIndex.h"

#include <exec/train/optimizer/Optimizer.h>

namespace onert
{
namespace backend
{
namespace train
{

// TODO Support dynamic tensors
class TensorBuilder
{
public:
  TensorBuilder(const std::shared_ptr<TensorRegistry> &tensor_reg,
                const exec::train::optimizer::Optimizer *optimizer, const std::string planner_id);

  /**
   * @brief     Register tensor information to allocate on train backend
   * @param[in] ind    Operand index
   * @param[in] info   Operand information
   * @param[in] layout Operand data layout
   */
  void registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                          ir::Layout backend_layout);

  /**
   * @brief     Register informations of tensor used only in backward to allocate on train backend
   * @param[in] ind    Operand index
   * @param[in] info   Operand information
   * @param[in] layout Operand data layout
   */
  void registerBackwardTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                                  ir::Layout backend_layout);

  void registerDisposableBackwardTensorInfo(const DisposableTensorIndex &index,
                                            const ir::OperandInfo &info, ir::Layout layout);

  // TODO Support memory plan of all tensors
  void notifyFirstUse(const ir::OperandIndex &);
  void notifyLastUse(const ir::OperandIndex &);
  void notifyBackwardFirstUse(const ir::OperandIndex &);
  void notifyDisposableBackPropFirstUse(const DisposableTensorIndex &);
  void notifyDisposableBackPropLastUse(const DisposableTensorIndex &);

  bool isRegistered(const ir::OperandIndex &) const;
  bool isRegisteredBackward(const ir::OperandIndex &) const;

  void allocate(void);
  void allocateBackward(void);

private:
  const std::shared_ptr<TensorRegistry> _tensor_reg;
  std::unique_ptr<TensorManager> _tensor_mgr;
  ir::OperandIndexMap<ir::OperandInfo> _tensor_info_map;
  ir::OperandIndexMap<ir::OperandInfo> _backward_tensor_info_map;
  ir::OperandIndexMap<bool> _as_constants;
  const exec::train::optimizer::Optimizer *_optimizer;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_TENSOR_BUILDER_H__
