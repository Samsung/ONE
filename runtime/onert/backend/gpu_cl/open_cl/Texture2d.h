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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_TEXTURE2D_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_TEXTURE2D_H__

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ClCommandQueue.h"
#include "ClContext.h"
#include "GpuObject.h"
#include "OpenclWrapper.h"
#include "TensorType.h"
#include "Util.h"
#include "DataType.h"
#include "Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

struct Texture2DDescriptor : public GPUObjectDescriptor
{
  DataType element_type;
  bool normalized = false;  // used with INT data types, if normalized, we read
                            // in kernel float data.
  DataType normalized_type; // can be FLOAT32 or FLOAT16, using with normalized
                            // = true

  // optional
  int2 size = int2(0, 0);
  std::vector<uint8_t> data;

  Texture2DDescriptor() = default;
  Texture2DDescriptor(const Texture2DDescriptor &) = default;
  Texture2DDescriptor &operator=(const Texture2DDescriptor &) = default;
  Texture2DDescriptor(Texture2DDescriptor &&desc);
  Texture2DDescriptor &operator=(Texture2DDescriptor &&desc);

  absl::Status PerformSelector(const std::string &selector, const std::vector<std::string> &args,
                               const std::vector<std::string> &template_args,
                               std::string *result) const override;

  GPUResources GetGPUResources() const override;
  absl::Status PerformReadSelector(const std::vector<std::string> &args, std::string *result) const;

  absl::Status CreateGPUObject(CLContext *context, GPUObjectPtr *result) const override;
  void Release() override;
};

// Texture2D represent formatted GPU data storage.
// Texture2D is moveable but not copyable.
class Texture2D : public GPUObject
{
public:
  Texture2D() {} // just for using Texture2D as a class members
  Texture2D(cl_mem texture, int width, int height, cl_channel_type type);

  // Move only
  Texture2D(Texture2D &&texture);
  Texture2D &operator=(Texture2D &&texture);
  Texture2D(const Texture2D &) = delete;
  Texture2D &operator=(const Texture2D &) = delete;

  virtual ~Texture2D() { Release(); }

  cl_mem GetMemoryPtr() const { return texture_; }

  // Writes data to a texture. Data should point to a region that
  // has exact width * height * sizeof(pixel) bytes.
  template <typename T> absl::Status WriteData(CLCommandQueue *queue, const absl::Span<T> data);

  // Reads data from Texture2D into CPU memory.
  template <typename T> absl::Status ReadData(CLCommandQueue *queue, std::vector<T> *result) const;

  absl::Status GetGPUResources(const GPUObjectDescriptor *obj_ptr,
                               GPUResourcesWithValue *resources) const override;

  absl::Status CreateFromTexture2DDescriptor(const Texture2DDescriptor &desc, CLContext *context);

private:
  void Release();

  cl_mem texture_ = nullptr;
  int width_;
  int height_;
  cl_channel_type channel_type_;
};

using Texture2DPtr = std::shared_ptr<Texture2D>;

// Creates new 4-channel 2D texture with f32 elements
absl::Status CreateTexture2DRGBA32F(int width, int height, CLContext *context, Texture2D *result);

// Creates new 4-channel 2D texture with f16 elements
absl::Status CreateTexture2DRGBA16F(int width, int height, CLContext *context, Texture2D *result);

absl::Status CreateTexture2DRGBA(DataType type, int width, int height, CLContext *context,
                                 Texture2D *result);

absl::Status CreateTexture2DRGBA(DataType type, int width, int height, void *data,
                                 CLContext *context, Texture2D *result);

template <typename T>
absl::Status Texture2D::WriteData(CLCommandQueue *queue, const absl::Span<T> data)
{
  const int element_size = ChannelTypeToSizeInBytes(channel_type_);
  if (sizeof(T) % element_size != 0)
  {
    return absl::InvalidArgumentError(
      "Template type T has not suitable element type for created texture.");
  }
  if (4 * width_ * height_ * element_size != data.size() * sizeof(T))
  {
    return absl::InvalidArgumentError(
      "absl::Span<T> data size is different from texture allocated size.");
  }

  RETURN_IF_ERROR(queue->EnqueueWriteImage(texture_, int3(width_, height_, 1), data.data()));

  return absl::OkStatus();
}

template <typename T>
absl::Status Texture2D::ReadData(CLCommandQueue *queue, std::vector<T> *result) const
{
  const int element_size = ChannelTypeToSizeInBytes(channel_type_);
  if (sizeof(T) != element_size)
  {
    return absl::InvalidArgumentError("Pixel format is different.");
  }

  const int elements_count = width_ * height_ * 4;
  result->resize(elements_count);

  return queue->EnqueueReadImage(texture_, int3(width_, height_, 1), result->data());
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_TEXTURE2D_H__
