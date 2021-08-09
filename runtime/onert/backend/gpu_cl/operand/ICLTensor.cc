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

  TensorFloat32 src_tensor;

  switch (_shape.rank())
  {
    case 1:
      src_tensor.shape = BHWC(_shape.dim(0), 1, 1, 1);
      break;
    case 2:
      src_tensor.shape = BHWC(_shape.dim(0), 1, 1, _shape.dim(1));
      break;
    case 3:
      src_tensor.shape = BHWC(_shape.dim(0), 1, _shape.dim(1), _shape.dim(2));
      break;
    case 4:
      src_tensor.shape = BHWC(_shape.dim(0), _shape.dim(1), _shape.dim(2), _shape.dim(3));
      break;
  }

  src_tensor.data = std::vector<float>((float *)ptr, (float *)ptr + _shape.num_elements());

  if (!handle()->WriteData(_queue, src_tensor).ok())
  {
    throw std::runtime_error("Failed to WriteData.");
  }
}

void ICLTensor::enqueueReadBuffer(void *ptr, bool)
{
  TensorFloat32 dst_tensor;

  switch (_shape.rank())
  {
    case 1:
      dst_tensor.shape = BHWC(_shape.dim(0), 1, 1, 1);
      break;
    case 2:
      dst_tensor.shape = BHWC(_shape.dim(0), 1, 1, _shape.dim(1));
      break;
    case 3:
      dst_tensor.shape = BHWC(_shape.dim(0), 1, _shape.dim(1), _shape.dim(2));
      break;
    case 4:
      dst_tensor.shape = BHWC(_shape.dim(0), _shape.dim(1), _shape.dim(2), _shape.dim(3));
      break;
  }
  dst_tensor.data = std::vector<float>(_shape.num_elements(), 0);

  if (!handle()->ReadData(_queue, &dst_tensor).ok())
  {
    throw std::runtime_error("Failed to ReadData.");
  }

  memcpy(ptr, dst_tensor.data.data(), total_size());
}

} // namespace operand
} // namespace gpu_cl
} // namespace backend
} // namespace onert
