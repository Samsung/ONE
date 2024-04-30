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

#ifndef __ONERT_BACKEND_TRAIN_TENSOR_MANAGER_H__
#define __ONERT_BACKEND_TRAIN_TENSOR_MANAGER_H__

#include "DisposableTensorIndex.h"
#include "MemoryManager.h"
#include "TensorRegistry.h"

#include <ir/OperandIndexMap.h>
#include <ir/OperandInfo.h>

namespace onert
{
namespace backend
{
namespace train
{

class TensorManager
{
public:
  // Minimum alignment for buffers
  //
  // _align is equal to EIGEN_MAX_ALIGN_BYTES, though including Eigen headers
  // here to get that symbol may not be a good idea.
  static constexpr uint64_t _align = 16;

public:
  TensorManager(const std::shared_ptr<TensorRegistry> &reg, const std::string planner_id);
  virtual ~TensorManager() = default;

  void allocateNonConstTensors();
  void allocateTrainableTensors();
  void allocateBackPropTensors();
  void allocateGradientTensors();
  void allocateDisposableBackPropTensors();
  // TODO Add member functions to deallocate tensors

  void claimNonConstPlan(const ir::OperandIndex &ind);
  void releaseNonConstPlan(const ir::OperandIndex &ind);
  void claimTrainablePlan(const ir::OperandIndex &ind);
  void releaseTrainablePlan(const ir::OperandIndex &ind);
  void claimBackPropPlan(const ir::OperandIndex &ind);
  void releaseBackPropPlan(const ir::OperandIndex &ind);
  void claimGradientPlan(const ir::OperandIndex &ind);
  void releaseGradientPlan(const ir::OperandIndex &ind);
  void claimDisposableBackPropPlan(const DisposableTensorIndex &ind);
  void releaseDisposableBackPropPlan(const DisposableTensorIndex &ind);

private:
  std::unique_ptr<MemoryManager> _nonconst_mgr;
  std::unique_ptr<MemoryManager> _trainable_mgr;
  std::unique_ptr<MemoryManager> _back_prop_mgr;
  std::unique_ptr<MemoryManager> _gradient_mgr;
  std::unique_ptr<DisposableMemoryManager> _disposable_back_prop_mgr;
  const std::shared_ptr<TensorRegistry> _tensors;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_TENSOR_MANAGER_H__
