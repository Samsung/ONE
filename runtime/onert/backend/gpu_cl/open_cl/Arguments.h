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

#ifndef __ONERT_BACKEND_GPU_CL_ARGUMENTS_H__
#define __ONERT_BACKEND_GPU_CL_ARGUMENTS_H__

#include <map>
#include <string>
#include <vector>

#include "ClDevice.h"
#include "GpuObject.h"
#include "OpenclWrapper.h"

#include "AccessType.h"
#include "Types.h"
#include "Util.h"
#include "Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class ArgumentsBinder
{
public:
  virtual absl::Status SetInt(const std::string &name, int value) = 0;
  virtual absl::Status SetFloat(const std::string &name, float value) = 0;
  virtual ~ArgumentsBinder() = default;
};

class Arguments : public ArgumentsBinder
{
public:
  Arguments() = default;
  void AddFloat(const std::string &name, float value = 0.0f);
  void AddInt(const std::string &name, int value = 0);
  void AddObjectRef(const std::string &name, AccessType access_type,
                    GPUObjectDescriptorPtr &&descriptor_ptr);
  void AddObject(const std::string &name, GPUObjectDescriptorPtr &&descriptor_ptr);

  absl::Status SetInt(const std::string &name, int value) override;
  absl::Status SetFloat(const std::string &name, float value) override;
  absl::Status SetObjectRef(const std::string &name, const GPUObject *object);

  absl::Status Bind(cl_kernel kernel, int offset = 0);

  void RenameArgs(const std::string &postfix, std::string *code) const;
  absl::Status Merge(Arguments &&args, const std::string &postfix);

  absl::Status AllocateObjects(CLContext *context);
  void ReleaseCPURepresentation();
  absl::Status TransformToCLCode(const DeviceInfo &device_info,
                                 const std::map<std::string, std::string> &linkables,
                                 std::string *code);

  // Move only
  Arguments(Arguments &&args);
  Arguments &operator=(Arguments &&args);
  Arguments(const Arguments &) = delete;
  Arguments &operator=(const Arguments &) = delete;

  ~Arguments() override = default;

private:
  void AddBuffer(const std::string &name, const GPUBufferDescriptor &desc);
  void AddImage2D(const std::string &name, const GPUImage2DDescriptor &desc);
  void AddImage2DArray(const std::string &name, const GPUImage2DArrayDescriptor &desc);
  void AddImage3D(const std::string &name, const GPUImage3DDescriptor &desc);
  void AddImageBuffer(const std::string &name, const GPUImageBufferDescriptor &desc);
  void AddCustomMemory(const std::string &name, const GPUCustomMemoryDescriptor &desc);

  absl::Status SetImage2D(const std::string &name, cl_mem memory);
  absl::Status SetBuffer(const std::string &name, cl_mem memory);
  absl::Status SetImage2DArray(const std::string &name, cl_mem memory);
  absl::Status SetImage3D(const std::string &name, cl_mem memory);
  absl::Status SetImageBuffer(const std::string &name, cl_mem memory);
  absl::Status SetCustomMemory(const std::string &name, cl_mem memory);

  std::string GetListOfArgs();

  std::string AddActiveArgument(const std::string &arg_name, bool use_f32_for_halfs);
  void AddGPUResources(const std::string &name, const GPUResources &resources);

  absl::Status SetGPUResources(const std::string &name, const GPUResourcesWithValue &resources);

  absl::Status AddObjectArgs();

  void ResolveArgsPass(const DeviceInfo &device_info, std::string *code);
  absl::Status ResolveSelectorsPass(const std::map<std::string, std::string> &linkables,
                                    std::string *code);

  absl::Status ResolveSelector(const std::map<std::string, std::string> &linkables,
                               const std::string &object_name, const std::string &selector,
                               const std::vector<std::string> &args,
                               const std::vector<std::string> &template_args, std::string *result);

  void ResolveObjectNames(const std::string &object_name,
                          const std::vector<std::string> &member_names, std::string *code);

  GPUObjectDescriptor *GetObjectDescriptor(const std::string &object_name) const;

  static constexpr char kArgsPrefix[] = "args.";

  struct IntValue
  {
    int value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, IntValue> int_values_;
  std::vector<int32_t> shared_int4s_data_;

  struct FloatValue
  {
    float value;

    // many uniforms generated automatically and not used
    // to reduce amount of data transferred we adding this optimization
    bool active = false;

    // offset to shared uniform storage.
    uint32_t offset = -1;
  };
  std::map<std::string, FloatValue> float_values_;
  std::vector<float> shared_float4s_data_;

  std::map<std::string, GPUBufferDescriptor> buffers_;
  std::map<std::string, GPUImage2DDescriptor> images2d_;
  std::map<std::string, GPUImage2DArrayDescriptor> image2d_arrays_;
  std::map<std::string, GPUImage3DDescriptor> images3d_;
  std::map<std::string, GPUImageBufferDescriptor> image_buffers_;
  std::map<std::string, GPUCustomMemoryDescriptor> custom_memories_;

  struct ObjectRefArg
  {
    GPUObjectDescriptorPtr descriptor;
  };
  std::map<std::string, ObjectRefArg> object_refs_;

  struct ObjectArg
  {
    GPUObjectPtr obj_ptr;
    GPUObjectDescriptorPtr descriptor;
  };
  std::map<std::string, ObjectArg> objects_;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_ARGUMENTS_H__
