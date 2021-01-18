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
#include <fstream>

#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/runtime/CL/CLTunerTypes.h>

#include "Config.h"

namespace onert
{
namespace backend
{
namespace acl_cl
{

Config::~Config()
{
  auto tuner_active = util::getConfigBool(util::config::TUNER_ACTIVE);
  auto tuner_update = util::getConfigBool(util::config::TUNER_UPDATE);
  auto tuner_filepath = util::getConfigString(util::config::TUNER_FILEPATH);

  if (_tuner != nullptr)
  {
    if (tuner_active == true && tuner_update == true)
    {
      _tuner->save_to_file(tuner_filepath);
    }
    delete _tuner;
    _tuner = nullptr;
  }
}

bool Config::file_exists(const std::string &file_name)
{
  std::ifstream file(file_name);
  return file.good();
}

bool Config::set_tuner()
{
  auto tuner_active = util::getConfigBool(util::config::TUNER_ACTIVE);
  auto tuner_mode = util::getConfigInt(util::config::TUNER_MODE);
  auto tuner_filepath = util::getConfigString(util::config::TUNER_FILEPATH);

  if (tuner_active)
  {
    _tuner = new arm_compute::CLTuner();

    if (tuner_mode <= CONFIG_CLTUNER_MIN || tuner_mode >= CONFIG_CLTUNER_MAX)
    {
      return false;
    }

    _tuner->set_tune_new_kernels(tuner_active);
    switch (tuner_mode)
    {
      case CONFIG_CLTUNER_READ:
        if (file_exists(tuner_filepath))
        {
          _tuner->load_from_file(tuner_filepath);
        }
        break;
      case CONFIG_CLTUNER_EXHAUSTIVE:
        _tuner->set_tuner_mode(arm_compute::CLTunerMode::EXHAUSTIVE);
        break;
      case CONFIG_CLTUNER_NORMAL:
        _tuner->set_tuner_mode(arm_compute::CLTunerMode::NORMAL);
        break;
      case CONFIG_CLTUNER_RAPID:
        _tuner->set_tuner_mode(arm_compute::CLTunerMode::RAPID);
        break;
      default:
        return false;
    }
  }

  return true;
}

bool Config::initialize()
{
  if (!arm_compute::opencl_is_available())
  {
    return false;
  }

  if (set_tuner() == false)
  {
    _tuner = nullptr;
  }

  arm_compute::CLScheduler::get().default_init(_tuner);

  // NOTE CLKernelLibraryEx must use the same context as CLScheduler
  // It did not check whether another device is available.
  arm_compute::CLKernelLibraryEx::get().init(
    "./cl_kernels/", arm_compute::CLScheduler::get().context(), cl::Device::getDefault());

  return true;
}

ir::Layout Config::supportLayout(const ir::Operation &, ir::Layout frontend_layout)
{
  const std::string acl_layout_str = util::getConfigString(util::config::ACL_LAYOUT);
  if (acl_layout_str == "NHWC")
  {
    return ir::Layout::NHWC;
  }
  else if (acl_layout_str == "NCHW")
  {
    return ir::Layout::NCHW;
  }

  return frontend_layout;
}

} // namespace acl_cl
} // namespace backend
} // namespace onert
