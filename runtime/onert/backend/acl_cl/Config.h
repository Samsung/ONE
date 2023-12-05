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

#ifndef __ONERT_BACKEND_ACL_CL_CONFIG_H__
#define __ONERT_BACKEND_ACL_CL_CONFIG_H__

#include "CLTimer.h"
#include <memory>
#include <backend/IConfig.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

namespace onert
{
namespace backend
{
namespace acl_cl
{

class Config : public IConfig
{
public:
  std::string id() override { return "acl_cl"; }
  bool initialize() override;
  bool supportDynamicTensor() override { return false; }
  bool supportFP16() override { return true; }
  void sync() const override { arm_compute::CLScheduler::get().sync(); }

  std::unique_ptr<util::ITimer> timer() override { return std::make_unique<CLTimer>(); }
};

} // namespace acl_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_CL_CONFIG_H__
