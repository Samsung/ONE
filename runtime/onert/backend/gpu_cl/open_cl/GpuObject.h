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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_GPU_OBJECT_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_GPU_OBJECT_H__

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ClContext.h"
#include "OpenclWrapper.h"
#include "AccessType.h"
#include "DataType.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

struct GPUImage2DDescriptor
{
  DataType data_type;
  AccessType access_type;
  cl_mem memory;
};

struct GPUImage3DDescriptor
{
  DataType data_type;
  AccessType access_type;
  cl_mem memory;
};

struct GPUImage2DArrayDescriptor
{
  DataType data_type;
  AccessType access_type;
  cl_mem memory;
};

struct GPUImageBufferDescriptor
{
  DataType data_type;
  AccessType access_type;
  cl_mem memory;
};

struct GPUCustomMemoryDescriptor
{
  std::string type_name;
  cl_mem memory;
};

enum class MemoryType
{
  GLOBAL,
  CONSTANT,
  LOCAL
};

std::string MemoryTypeToCLType(MemoryType type);

struct GPUBufferDescriptor
{
  DataType data_type;
  AccessType access_type;
  int element_size;
  MemoryType memory_type = MemoryType::GLOBAL;
  std::vector<std::string> attributes;
  cl_mem memory;
};

struct GPUResources
{
  std::vector<std::string> ints;
  std::vector<std::string> floats;
  std::vector<std::pair<std::string, GPUBufferDescriptor>> buffers;
  std::vector<std::pair<std::string, GPUImage2DDescriptor>> images2d;
  std::vector<std::pair<std::string, GPUImage2DArrayDescriptor>> image2d_arrays;
  std::vector<std::pair<std::string, GPUImage3DDescriptor>> images3d;
  std::vector<std::pair<std::string, GPUImageBufferDescriptor>> image_buffers;
  std::vector<std::pair<std::string, GPUCustomMemoryDescriptor>> custom_memories;

  std::vector<std::string> GetNames() const
  {
    std::vector<std::string> names = ints;
    names.insert(names.end(), floats.begin(), floats.end());
    for (const auto &obj : buffers)
    {
      names.push_back(obj.first);
    }
    for (const auto &obj : images2d)
    {
      names.push_back(obj.first);
    }
    for (const auto &obj : image2d_arrays)
    {
      names.push_back(obj.first);
    }
    for (const auto &obj : images3d)
    {
      names.push_back(obj.first);
    }
    for (const auto &obj : image_buffers)
    {
      names.push_back(obj.first);
    }
    for (const auto &obj : custom_memories)
    {
      names.push_back(obj.first);
    }
    return names;
  }
};

struct GPUResourcesWithValue
{
  std::vector<std::pair<std::string, int>> ints;
  std::vector<std::pair<std::string, float>> floats;
  std::vector<std::pair<std::string, cl_mem>> buffers;
  std::vector<std::pair<std::string, cl_mem>> images2d;
  std::vector<std::pair<std::string, cl_mem>> image2d_arrays;
  std::vector<std::pair<std::string, cl_mem>> images3d;
  std::vector<std::pair<std::string, cl_mem>> image_buffers;
  std::vector<std::pair<std::string, cl_mem>> custom_memories;
};

class GPUObject;

class GPUObjectDescriptor
{
public:
  GPUObjectDescriptor() = default;
  GPUObjectDescriptor(const GPUObjectDescriptor &) = default;
  GPUObjectDescriptor &operator=(const GPUObjectDescriptor &) = default;
  GPUObjectDescriptor(GPUObjectDescriptor &&obj_desc) : state_vars_(std::move(obj_desc.state_vars_))
  {
  }
  GPUObjectDescriptor &operator=(GPUObjectDescriptor &&obj_desc)
  {
    if (this != &obj_desc)
    {
      state_vars_ = std::move(obj_desc.state_vars_);
    }
    return *this;
  }
  virtual ~GPUObjectDescriptor() = default;

  void SetStateVar(const std::string &key, const std::string &value) const
  {
    state_vars_[key] = value;
  }

  virtual std::string PerformConstExpr(const std::string &) const { return ""; }

  virtual absl::Status PerformSelector(const std::string &, const std::vector<std::string> &,
                                       const std::vector<std::string> &, std::string *result) const
  {
    *result = "";
    return absl::OkStatus();
  }
  virtual GPUResources GetGPUResources() const { return GPUResources(); }

  virtual absl::Status CreateGPUObject(CLContext *, std::unique_ptr<GPUObject> *) const
  {
    return absl::OkStatus();
  }
  virtual void Release() {}

  void SetAccess(AccessType access_type) { access_type_ = access_type; }
  AccessType GetAccess() const { return access_type_; }

protected:
  // friend flatbuffers::Offset<data::GPUObjectDescriptor> Encode(
  //     const GPUObjectDescriptor& desc, flatbuffers::FlatBufferBuilder* builder);
  // friend void Decode(const data::GPUObjectDescriptor* fb_obj,
  //                    GPUObjectDescriptor* obj);
  mutable std::map<std::string, std::string> state_vars_;
  AccessType access_type_;
};

using GPUObjectDescriptorPtr = std::unique_ptr<GPUObjectDescriptor>;

class GPUObject
{
public:
  GPUObject() = default;
  // Move only
  GPUObject(GPUObject &&obj_desc) = default;
  GPUObject &operator=(GPUObject &&obj_desc) = default;
  GPUObject(const GPUObject &) = delete;
  GPUObject &operator=(const GPUObject &) = delete;
  virtual ~GPUObject() = default;
  virtual absl::Status GetGPUResources(const GPUObjectDescriptor *obj_ptr,
                                       GPUResourcesWithValue *resources) const = 0;
};

using GPUObjectPtr = std::unique_ptr<GPUObject>;

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_GPU_OBJECT_H__
