/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "SimpleSelectors.h"

#include <memory>
#include <set>

#include "open_cl/kernels/Add.h"
#include "open_cl/kernels/Softmax.h"
#include "open_cl/kernels/Softmax1x1.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

void SelectAdd(const OperationDef &op_def, const std::vector<int> &channels, int dst_channels,
               std::unique_ptr<GPUOperation> *ptr)
{
  GPUOperation operation = CreateAdd(op_def, channels, dst_channels);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

void SelectSoftmax(const BHWC &shape, const OperationDef &op_def,
                   std::unique_ptr<GPUOperation> *ptr)
{
  if (shape.w == 1 && shape.h == 1)
  {
    Softmax1x1 operation = CreateSoftmax1x1(op_def);
    *ptr = absl::make_unique<Softmax1x1>(std::move(operation));
  }
  else
  {
    GPUOperation operation = CreateSoftmax(op_def);
    *ptr = absl::make_unique<GPUOperation>(std::move(operation));
  }
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
