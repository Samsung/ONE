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

#include "ICLTensor.h"

#include "open_cl/OpenclWrapper.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace operand
{



void ICLTensor::access(const std::function<void(ITensor &tensor)> &fn)
{
  if (total_size() == 0)
    return;

  fn(*this);
}

void ICLTensor::enqueueWriteBuffer(const void *ptr, bool)
{
  switch (handle()->GetStorageType())
  {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      if (!_queue->EnqueueWriteBuffer(handle()->GetMemoryPtr(), total_size(), ptr).ok())
      {
        return;
      }
      break;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:

    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      // TODO
      // Change int3 {1, 1, 1} to CalculateTextureRegion.
      if (!_queue->EnqueueWriteImage(handle()->GetMemoryPtr(), int3{1, 1, 1}, ptr).ok())
      {
        return;
      }
      break;
    default:
      break;
  }
}

void ICLTensor::enqueueReadBuffer(void *ptr, bool)
{
  switch (handle()->GetStorageType())
  {
    case TensorStorageType::BUFFER:
    case TensorStorageType::IMAGE_BUFFER:
      if (!_queue->EnqueueReadBuffer(handle()->GetMemoryPtr(), total_size(), ptr).ok())
      {
        return;
      }
      break;
    case TensorStorageType::TEXTURE_ARRAY:
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::TEXTURE_3D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      if (!_queue->EnqueueReadImage(handle()->GetMemoryPtr(), int3{1, 1, 1}, ptr).ok())
      {
        return;
      }
      break;
    default:
      break;
  }
}

} // namespace operand
} // namespace gpu_cl
} // namespace backend
} // namespace onert
