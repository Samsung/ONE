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

#include "CustomKernelRegistry.h"

#include <memory>

namespace onert
{
namespace frontend
{
namespace custom
{

void KernelRegistry::registerKernel(const std::string &id, nnfw_custom_eval evalFunction)
{
  _storage.emplace(id, evalFunction);
}

std::shared_ptr<backend::custom::IKernelBuilder> KernelRegistry::getBuilder()
{
  return std::make_unique<KernelBuilder>(this);
}

std::unique_ptr<Kernel> KernelRegistry::buildKernelForOp(const std::string &id)
{
  auto it = _storage.find(id);
  if (it == _storage.end())
  {
    throw std::runtime_error("Unable to find associated kernel for op");
  }

  return std::make_unique<Kernel>(it->second);
}

// Kernel builder
std::unique_ptr<exec::IFunction>
KernelBuilder::buildKernel(const std::string &id,
                           backend::custom::CustomKernelConfigParams &&params) const
{
  auto kernel = _registry->buildKernelForOp(id);
  kernel->configure(std::move(params));

  return kernel;
}

KernelBuilder::KernelBuilder(KernelRegistry *registry) : _registry(registry) {}

} // namespace custom
} // namespace frontend
} // namespace onert
