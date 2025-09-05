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

#include "DisposableTensorIndex.h"
#include "LayerScopeTensorIndex.h"
#include "TensorManager.h"
#include "TensorRegistry.h"
#include "util/Set.h"

#include <exec/train/optimizer/Optimizer.h>
#include <ir/OperationIndexMap.h>

namespace onert::backend::train
{

// TODO Support dynamic tensors
class TensorBuilder
{
public:
  TensorBuilder(const std::shared_ptr<TensorRegistry> &tensor_reg,
                const exec::train::optimizer::Optimizer *optimizer);

  /**
   * @brief     Register tensor information to allocate on train backend
   * @param[in] ind    Operand index
   * @param[in] info   Operand information
   */
  void registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info);

  /**
   * @brief     Register informations of tensor used only in backward to allocate on train backend
   * @param[in] ind    Operand index
   * @param[in] info   Operand information
   */
  void registerBackwardTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info);

  void registerDisposableBackwardTensorInfo(const DisposableTensorIndex &index,
                                            const ir::OperandInfo &info);

  void registerLayerScopeTensor(const LayerScopeTensorIndex &index,
                                std::shared_ptr<LayerScopeTensor> &info);

  // TODO Support memory plan of all tensors
  void notifyFirstUse(const ir::OperandIndex &);
  void notifyLastUse(const ir::OperandIndex &);
  void notifyBackwardFirstUse(const ir::OperandIndex &);
  void notifyBackwardLastUse(const ir::OperandIndex &);
  void notifyDisposableBackPropFirstUse(const DisposableTensorIndex &);
  void notifyDisposableBackPropLastUse(const DisposableTensorIndex &);
  void notifyLayerScopeFirstUse(const LayerScopeTensorIndex &);
  void notifyLayerScopeLastUse(const LayerScopeTensorIndex &);

  bool isRegistered(const ir::OperandIndex &) const;
  bool isRegisteredBackward(const ir::OperandIndex &) const;
  bool isRegisteredDisposableBackwardTensor(const DisposableTensorIndex &index) const;
  bool isRegisteredLayerScopeTensor(const ir::OperationIndex &) const;

  const util::Set<LayerScopeTensorIndex> &
  getRegisteredLayerScopeTensorIndices(const ir::OperationIndex &) const;
  LayerScopeTensorLifeTime getLayerScopeTensorLifeTime(const LayerScopeTensorIndex &) const;

  void allocate(void);
  void allocateBackward(void);
  void allocateLayerScope(void);

private:
  const std::shared_ptr<TensorRegistry> _tensor_reg;
  std::unique_ptr<TensorManager> _tensor_mgr;
  ir::OperandIndexMap<ir::OperandInfo> _tensor_info_map;
  ir::OperandIndexMap<ir::OperandInfo> _backward_tensor_info_map;
  ir::OperandIndexMap<bool> _as_constants;
  util::Set<DisposableTensorIndex> _disposable_backprops;
  ir::OperationIndexMap<util::Set<LayerScopeTensorIndex>> _operation_to_layerscope;
  const exec::train::optimizer::Optimizer *_optimizer;
};

} // namespace onert::backend::train

#endif // __ONERT_BACKEND_TRAIN_TENSOR_BUILDER_H__
