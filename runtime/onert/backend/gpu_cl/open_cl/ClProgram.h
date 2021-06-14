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

#ifndef __ONERT_BACKEND_GPU_CL_CL_PROGRAM_H__
#define __ONERT_BACKEND_GPU_CL_CL_PROGRAM_H__

#include <cstdint>
#include <vector>

#include "ClContext.h"
#include "ClDevice.h"
#include "OpenclWrapper.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

enum class CompilerOptions
{
  // ADRENO_FULL_SIMD_LINE:
  //   Adreno can have 2 sizes for SIMD size.
  //   On Adreno 4xx/5xx it is 32/64, on Adreno6xx it is 64/128.
  //   Some our algorithms actually rely on exact size, for example on full
  //   SIMD size, so we need this define.
  //   This define is actually -qcom-accelerate-16-bit, but it controls SIMD
  //   size.
  ADRENO_FULL_SIMD_LINE,
  ADRENO_MORE_WAVES,
  POWERVR_FP16,
  CL_OPT_DISABLE,
  CL_2_0,
  CL_3_0,
};

std::string CompilerOptionsToString(const CLDevice &device,
                                    const std::vector<CompilerOptions> &compiler_options);

class CLProgram
{
public:
  CLProgram() {}
  CLProgram(cl_program program, cl_device_id device_id);

  // Move only
  CLProgram(CLProgram &&program);
  CLProgram &operator=(CLProgram &&program);
  CLProgram(const CLProgram &) = delete;
  CLProgram &operator=(const CLProgram &) = delete;

  ~CLProgram();

  cl_program program() const { return program_; }

  // Return the cl_device_id associated with the program object.
  // This can be the device associated with context on which the program object
  // has been created or can be device that was specified when a program object
  // was created using clCreateProgramWithBinary.
  cl_device_id GetDeviceId() const { return device_id_; }

  absl::Status GetBinary(std::vector<uint8_t> *result) const;

private:
  void Release();

  cl_program program_ = nullptr;

  // reference
  cl_device_id device_id_ = nullptr;
};

absl::Status CreateCLProgram(const std::string &code, const std::string &compiler_options,
                             const CLContext &context, const CLDevice &device, CLProgram *result);

absl::Status CreateCLProgramFromBinary(const CLContext &context, const CLDevice &device,
                                       absl::Span<const uint8_t> binary, CLProgram *result);

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_CL_PROGRAM_H__
