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

#include "CLTensor.h"

#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type_util.h"

using namespace tflite::gpu::cl;

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace operand
{

CLTensor::CLTensor(size_t rank, TensorType type, tflite::gpu::BHWC shape,
                   tflite::gpu::TensorDescriptor desc)
  : ICLTensor{rank, type, shape, desc}, _tensor(std::make_shared<Tensor>())
{
}

const tflite::gpu::cl::Tensor *CLTensor::handle() const { return _tensor.get(); }

tflite::gpu::cl::Tensor *CLTensor::handle() { return _tensor.get(); }

} // namespace operand
} // namespace gpu_cl
} // namespace backend
} // namespace onert
