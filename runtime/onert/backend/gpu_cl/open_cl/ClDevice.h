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

#ifndef __ONERT_BACKEND_GPU_CL_CL_DEVICE_H__
#define __ONERT_BACKEND_GPU_CL_CL_DEVICE_H__

#include <string>
#include <vector>

#include "DeviceInfo.h"
#include "OpenclWrapper.h"
#include "Util.h"
#include "Types.h"
#include "Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

// A wrapper around opencl device id
class CLDevice
{
public:
  CLDevice() = default;
  CLDevice(cl_device_id id, cl_platform_id platform_id);

  CLDevice(CLDevice &&device);
  CLDevice &operator=(CLDevice &&device);
  CLDevice(const CLDevice &);
  CLDevice &operator=(const CLDevice &);

  ~CLDevice() {}

  cl_device_id id() const { return id_; }
  cl_platform_id platform() const { return platform_id_; }
  std::string GetPlatformVersion() const;

  Vendor vendor() const { return info_.vendor; }
  OpenCLVersion cl_version() const { return info_.cl_version; }
  bool SupportsFP16() const;
  bool SupportsTextureArray() const;
  bool SupportsImageBuffer() const;
  bool SupportsImage3D() const;
  bool SupportsExtension(const std::string &extension) const;
  bool SupportsFP32RTN() const;
  bool SupportsFP16RTN() const;
  bool IsCL20OrHigher() const;
  bool SupportsSubGroupWithSize(int sub_group_size) const;
  bool IsAdreno() const;
  bool IsAdreno3xx() const;
  bool IsAdreno4xx() const;
  bool IsAdreno5xx() const;
  bool IsAdreno6xx() const;
  bool IsAdreno6xxOrHigher() const;
  bool IsPowerVR() const;
  bool IsNvidia() const;
  bool IsMali() const;
  bool IsAMD() const;
  bool IsIntel() const;

  // To track bug on some Adreno. b/131099086
  bool SupportsOneLayerTextureArray() const;
  void DisableOneLayerTextureArray();

  const DeviceInfo &GetInfo() const { return info_; }
  // We update device info during context creation, so as supported texture
  // formats can be requested from context only.
  mutable DeviceInfo info_;

private:
  cl_device_id id_ = nullptr;
  cl_platform_id platform_id_ = nullptr;
};

absl::Status CreateDefaultGPUDevice(CLDevice *result);

template <typename T> T GetDeviceInfo(cl_device_id id, cl_device_info info)
{
  T result;
  cl_int error = clGetDeviceInfo(id, info, sizeof(T), &result, nullptr);
  if (error != CL_SUCCESS)
  {
    return -1;
  }
  return result;
}

template <typename T> absl::Status GetDeviceInfo(cl_device_id id, cl_device_info info, T *result)
{
  cl_int error = clGetDeviceInfo(id, info, sizeof(T), result, nullptr);
  if (error != CL_SUCCESS)
  {
    return absl::InvalidArgumentError(CLErrorCodeToString(error));
  }
  return absl::OkStatus();
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_CL_DEVICE_H__
