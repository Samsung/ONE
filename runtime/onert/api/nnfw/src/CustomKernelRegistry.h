/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_API_CUSTOM_KERNEL_REGISTRY_H__
#define __ONERT_API_CUSTOM_KERNEL_REGISTRY_H__

#include "CustomKernel.h"

#include <unordered_map>
#include <functional>
#include <memory>

#include <iostream>

namespace onert
{
namespace api
{

class CustomKernelRegistry
{
public:
  void registerKernel(const std::string &id, nnfw_custom_eval evalFunction);

  std::shared_ptr<backend::custom::IKernelBuilder> getBuilder();
  std::unique_ptr<CustomKernel> buildKernelForOp(const std::string &id);

private:
  std::unordered_map<std::string, nnfw_custom_eval> _storage;
};

} // namespace api
} // namespace onert

#endif // __ONERT_API_CUSTOM_KERNEL_REGISTRY_H__
