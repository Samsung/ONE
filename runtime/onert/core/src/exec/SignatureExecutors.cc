/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SignatureExecutors.h"

namespace onert::exec
{

SignatureExecutors::SignatureExecutors(const std::shared_ptr<IExecutors> &executors,
                                       const std::string &signature,
                                       const ir::SubgraphIndex &entry_index)
  : _executors(executors), _signature(signature), _entry_index(entry_index)
{
  // Check single model
  // TODO Support multimodel
  assert(dynamic_cast<SingleModelExecutors *>(executors.get()) != nullptr);
}

IExecutor *SignatureExecutors::entryExecutor() const
{
  return _executors->at(ir::ModelIndex{0}, _entry_index);
}

} // namespace onert::exec
