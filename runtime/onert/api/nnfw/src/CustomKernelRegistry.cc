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
namespace api
{

class KernelBuilder : public backend::custom::IKernelBuilder
{
public:
  KernelBuilder(CustomKernelRegistry *registry) : _registry(registry) {}

  std::unique_ptr<exec::IFunction>
  buildKernel(const std::string &id,
              backend::custom::CustomKernelConfigParams &&params) const override
  {
    auto kernel = _registry->buildKernelForOp(id);
    kernel->configure(std::move(params));

    return kernel;
  }

private:
  CustomKernelRegistry *_registry;
};

void CustomKernelRegistry::registerKernel(const std::string &id, nnfw_custom_eval evalFunction)
{
  _storage.emplace(id, evalFunction);
}

std::shared_ptr<backend::custom::IKernelBuilder> CustomKernelRegistry::getBuilder()
{
  return std::make_unique<KernelBuilder>(this);
}

std::unique_ptr<CustomKernel> CustomKernelRegistry::buildKernelForOp(const std::string &id)
{
  auto it = _storage.find(id);
  if (it == _storage.end())
  {
    throw std::runtime_error("Unable to find associated kernel for op");
  }

  return std::make_unique<CustomKernel>(it->second);
}

} // namespace api
} // namespace onert
