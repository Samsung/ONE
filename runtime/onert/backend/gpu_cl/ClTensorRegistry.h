/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_GPU_CL_TENSOR_REGISTRY_H__
#define __ONERT_BACKEND_GPU_CL_TENSOR_REGISTRY_H__

#include "backend/ITensorRegistry.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

/**
 * @brief Tensor registry class for acl backends
 *
 * This is implemented as a wrapper of AclTensorManager.
 */
template <typename T_ClTensorManager> class ClTensorRegistry : public ITensorRegistry
{
public:
  ClTensorRegistry(T_ClTensorManager *tensor_mgr) : _tensor_mgr{tensor_mgr} {}

  ITensor *getITensor(const ir::OperandIndex &ind) override { return _tensor_mgr->at(ind).get(); }

  ITensor *getNativeITensor(const ir::OperandIndex &ind) override { return getITensor(ind); }

  auto getClTensor(const ir::OperandIndex &ind) { return _tensor_mgr->at(ind).get(); }

  auto getClTensorReserver(const ir::OperandIndex &ind) { return _tensor_mgr->atR(ind); }

private:
  T_ClTensorManager *_tensor_mgr;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_TENSOR_REGISTRY_H__
