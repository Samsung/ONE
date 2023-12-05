/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

// For CLKernelLibraryEx initialization
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLKernelLibraryEx.h"

#include <util/ConfigSource.h>

#include <arm_compute/runtime/CL/CLScheduler.h>

#include "Config.h"

namespace onert
{
namespace backend
{
namespace acl_cl
{

bool Config::initialize()
{
  if (!arm_compute::opencl_is_available())
  {
    return false;
  }
  arm_compute::CLScheduler::get().default_init();
  // NOTE CLKernelLibraryEx must use the same context as CLScheduler
  // It did not check whether another device is available.
  arm_compute::CLKernelLibraryEx::get().init(
    "./cl_kernels/", arm_compute::CLScheduler::get().context(), cl::Device::getDefault());

  return true;
}

} // namespace acl_cl
} // namespace backend
} // namespace onert
