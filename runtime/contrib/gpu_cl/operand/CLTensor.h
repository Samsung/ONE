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

#ifndef __ONERT_BACKEND_GPU_CL_OPERAND_CL_TENSOR_H__
#define __ONERT_BACKEND_GPU_CL_OPERAND_CL_TENSOR_H__

#include "ICLTensor.h"

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace operand
{

class CLTensor : public ICLTensor
{
public:
  CLTensor() = delete;

public:
  CLTensor(size_t rank, TensorType type, tflite::gpu::BHWC shape,
           tflite::gpu::TensorDescriptor desc);

public:
  const tflite::gpu::cl::Tensor *handle() const override;
  tflite::gpu::cl::Tensor *handle() override;

public:
  /** Set given buffer as the buffer of the tensor
   *
   * @note Ownership of the memory is not transferred to this object.
   *       Thus management (allocate/free) should be done by the client.
   *
   * @param[in] host_ptr Storage to be used.
   */
  void setBuffer(void *host_ptr);

private:
  std::shared_ptr<tflite::gpu::cl::Tensor> _tensor;
};

} // namespace operand
} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPERAND_CL_TENSOR_H__
