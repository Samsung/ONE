
/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_EXEC_SINGLE_MODEL_EXECUTORS_H__
#define __ONERT_EXEC_SINGLE_MODEL_EXECUTORS_H__

#include "exec/IExecutors.h"

namespace onert
{
namespace exec
{

class SingleModelExecutors final : public IExecutors
{
public:
  SingleModelExecutors(void) = default;
  SingleModelExecutors(const SingleModelExecutors &) = delete;
  SingleModelExecutors(SingleModelExecutors &&) = default;

public:
  void emplace(const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index,
               std::unique_ptr<IExecutor> exec);

  IExecutor *at(const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index) const;

  uint32_t inputSize() const;

  uint32_t outputSize() const;

  const ir::OperandInfo inputInfo(const ir::IOIndex &index);

  const ir::OperandInfo outputInfo(const ir::IOIndex &index);

  void execute(const IODescription &desc);

private:
  // NOTE These executors does not have duplicated subgraph. This mean they do not allow support
  // subgraphs being called recursively because data of non-constant tensor of parent executor will
  // be updated by child executor. If you want to support subgraphs being called recursively, you
  // have to add allocate non-constant tensor memory of executors in execution time when each
  // subgraph is called.
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<IExecutor>> _executors;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_SINGLE_MODEL_EXECUTORS_H__
