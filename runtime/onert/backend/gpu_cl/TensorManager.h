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

#ifndef __ONERT_BACKEND_GPU_CL_TENSOR_MANAGER_H__
#define __ONERT_BACKEND_GPU_CL_TENSOR_MANAGER_H__

#include "MemoryManager.h"

#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"

#include "ir/OperandInfo.h"
#include "ir/OperandIndexMap.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class TensorManager
{
public:
  TensorManager(MemoryManager *const_mgr, MemoryManager *nonconst_mgr);

  virtual ~TensorManager() = default;

  void allocateConsts(void);
  void allocateNonconsts(void);
  void deallocateConsts(void);
  void deallocateNonconsts(void);

  void buildTensor(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                   tflite::gpu::cl::InferenceContext::CreateInferenceInfo create_info,
                   std::shared_ptr<tflite::gpu::cl::Environment> environment,
                   tflite::gpu::cl::DeviceInfo &device_info, TensorType type);

  std::shared_ptr<operand::ICLTensor> findTensorAsParent(const ir::OperandIndex &ind);

  void startLifetime(const ir::OperandIndex &ind);
  void finishLifetime(const ir::OperandIndex &ind);

  std::shared_ptr<operand::ICLTensor> at(const ir::OperandIndex &ind);
  std::shared_ptr<InferenceContextEx::DummyTensor> atR(const ir::OperandIndex &ind);

  InferenceContextEx::TensorReserverEx &constTensorReservers(void);
  InferenceContextEx::TensorReserverEx &nonconstTensorReservers(void);

  ir::OperandIndexMap<std::shared_ptr<operand::CLTensor>> &constTensors(void);
  ir::OperandIndexMap<std::shared_ptr<operand::CLTensor>> &nonconstTensors(void);

  void iterate(const std::function<void(const ir::OperandIndex &)> &fn);

  void tryDeallocConstants(void);

private:
  std::unique_ptr<MemoryManager> _const_mgr;
  std::unique_ptr<MemoryManager> _nonconst_mgr;
  ir::OperandIndexMap<MemoryManager &> _ind_to_mgr;
};

inline TensorManager *createTensorManager(tflite::gpu::cl::CLContext *context)
{
  VERBOSE(createTensorManager) << "GPU-CL TensorManager" << std::endl;
  return new TensorManager(new MemoryManager(context), new MemoryManager(context));
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_TENSOR_MANAGER_H__
