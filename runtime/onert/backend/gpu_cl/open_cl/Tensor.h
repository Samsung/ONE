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

#ifndef __ONERT_BACKEND_GPU_CL_TENSOR_H__
#define __ONERT_BACKEND_GPU_CL_TENSOR_H__

#include <cstdint>
#include <memory>

#include "absl/types/span.h"
#include "ClCommandQueue.h"
#include "OpenclWrapper.h"
#include "ClContext.h"
#include "ClDevice.h"
#include "ClMemory.h"
#include "GpuObject.h"
#include "TensorType.h"
#include "Util.h"
#include "DataType.h"
#include "Shape.h"
#include "Status.h"
#include "InternalTensor.h"
#include "Types.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class Tensor : public GPUObject
{
public:
  Tensor() : memory_(nullptr), image_buffer_memory_(nullptr), memory_owner_(true) {}
  Tensor(cl_mem memory, bool memory_owner, const BHWC &shape, const TensorDescriptor &descriptor);
  Tensor(cl_mem memory, bool memory_owner, const BHWDC &shape, const TensorDescriptor &descriptor);
  Tensor(cl_mem memory, bool memory_owner, cl_mem image_buffer_memory, const BHWC &shape,
         const TensorDescriptor &descriptor);
  Tensor(cl_mem memory, bool memory_owner, cl_mem image_buffer_memory, const BHWDC &shape,
         const TensorDescriptor &descriptor);

  // Move only
  Tensor(Tensor &&tensor);
  Tensor &operator=(Tensor &&tensor);
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  virtual ~Tensor() { Release(); }

  absl::Status GetGPUResources(const GPUObjectDescriptor *obj_ptr,
                               GPUResourcesWithValue *resources) const override;

  int Width() const { return shape_.w; }
  int Height() const { return shape_.h; }
  int Depth() const { return shape_.d; }
  int Channels() const { return shape_.c; }
  int Slices() const { return DivideRoundUp(shape_.c, 4); }
  int Batch() const { return shape_.b; }
  int3 GetFullTensorRegion() const;
  TensorDescriptor GetDescriptor() const { return descriptor_; }
  DataType GetDataType() const { return descriptor_.data_type; }
  TensorStorageType GetStorageType() const { return descriptor_.storage_type; }

  // for profiling and memory statistics
  uint64_t GetMemorySizeInBytes() const;

  cl_mem GetMemoryPtr() const;

  // This function returns buffer memory ptr for IMAGE_BUFFER instead of image
  // memory ptr.
  cl_mem GetMemoryPtrForWriting() const;

  absl::Status CreateFromDescriptor(const TensorDescriptor &desc, CLContext *context);

private:
  absl::Status IsValid(const BHWC &shape) const;
  absl::Status IsValid(const BHWDC &shape) const;

  int GetChannelsAlignment() const;
  int GetAlignedChannels() const;

  void Release();

  cl_mem memory_;
  cl_mem image_buffer_memory_; // for TensorStorageType::IMAGE_BUFFER only
  bool memory_owner_;
  BHWDC shape_;
  TensorDescriptor descriptor_;
};

using TensorPtr = std::shared_ptr<Tensor>;

absl::Status AllocateTensorMemory(const CLContext &context, const BHWC &shape,
                                  const TensorDescriptor &descriptor, CLMemory *result);

absl::Status AllocateTensorMemory(const CLContext &context, const BHWDC &shape,
                                  const TensorDescriptor &descriptor, CLMemory *result);

absl::Status CreateTensor(const CLContext &context, const BHWC &shape,
                          const TensorDescriptor &descriptor, Tensor *result);

absl::Status CreateTensor(const CLContext &context, const BHWDC &shape,
                          const TensorDescriptor &descriptor, Tensor *result);

absl::Status CreateSharedTensor(const CLContext &context, cl_mem memory, const BHWC &shape,
                                const TensorDescriptor &descriptor, Tensor *result);

absl::Status CreateSharedTensor(const CLContext &context, cl_mem memory, const BHWDC &shape,
                                const TensorDescriptor &descriptor, Tensor *result);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_CL_TENSOR_H__
