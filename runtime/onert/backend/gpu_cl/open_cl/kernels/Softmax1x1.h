/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_SOFTMAX1X1_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_SOFTMAX1X1_H_

#include "GpuOperation.h"

#include "open_cl/Precision.h"
#include "open_cl/ClKernel.h"
#include "open_cl/Tensor.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class Softmax1x1 : public GPUOperation {
 public:
  Softmax1x1() = default;
  explicit Softmax1x1(const OperationDef& definition);

  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  Softmax1x1(Softmax1x1&& kernel);
  Softmax1x1& operator=(Softmax1x1&& kernel);
  Softmax1x1(const Softmax1x1&) = delete;
  Softmax1x1& operator=(const Softmax1x1&) = delete;

  friend Softmax1x1 CreateSoftmax1x1();

 private:
  std::string GetSoftmaxKernelCode(const OperationDef& op_def);
};

Softmax1x1 CreateSoftmax1x1(const OperationDef& definition);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_KERNELS_SOFTMAX1X1_H_
