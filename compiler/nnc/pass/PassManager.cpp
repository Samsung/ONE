/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "pass/PassManager.h"
#include "pass/Pass.h"

namespace nnc
{

PassManager::PassManager() = default;

PassManager::~PassManager()
{
  for (auto &pass : _passes)
    pass->cleanup();
}

void PassManager::registerPass(std::unique_ptr<Pass> pass)
{
  _passes.push_back(std::move(pass));
} // registerPass

} // namespace nnc
