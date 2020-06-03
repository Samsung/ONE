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

#ifndef __ONERT_BACKEND_ACL_COMMON_KERNEL_ACL_FUNCTION_H__
#define __ONERT_BACKEND_ACL_COMMON_KERNEL_ACL_FUNCTION_H__

#include <exec/IFunction.h>
#include <arm_compute/runtime/IFunction.h>
#include <memory>

namespace onert
{
namespace backend
{
namespace acl_common
{

class AclFunction : public ::onert::exec::IFunction
{
public:
  AclFunction() = delete;

public:
  AclFunction(std::unique_ptr<::arm_compute::IFunction> &&func) : _func(std::move(func))
  {
    // DO NOTHING
  }

public:
  void run() override { _func->run(); }
  void runSync() override { run(); }
  void prepare() override { _func->prepare(); }

private:
  std::unique_ptr<::arm_compute::IFunction> _func;
};

class AclClFunction : public AclFunction
{
public:
  using AclFunction::AclFunction;

public:
  void runSync() final { run(); }
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_KERNEL_ACL_FUNCTION_H__
