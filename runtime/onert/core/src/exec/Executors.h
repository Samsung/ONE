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

#ifndef __ONERT_EXEC_EXECUTORS_H__
#define __ONERT_EXEC_EXECUTORS_H__

#include "exec/IExecutors.h"
#include "ir/NNPkg.h"

namespace std
{

template <> struct hash<std::pair<::onert::ir::ModelIndex, ::onert::ir::SubgraphIndex>>
{
  size_t
  operator()(const std::pair<::onert::ir::ModelIndex, ::onert::ir::SubgraphIndex> &pair) const
    noexcept
  {
    return (hash<uint32_t>()(pair.first.value()) << 16) ^ hash<uint32_t>()(pair.second.value());
  }
};

} // namespace std

namespace onert
{
namespace exec
{

/**
 * @brief Class to gather executors
 */
class Executors : public IExecutors
{
public:
  Executors(void) = default;
  Executors(std::unique_ptr<ir::ModelEdges> model_edges) { _model_edges = std::move(model_edges); }
  Executors(const Executors &) = delete;
  Executors(Executors &&) = default;
  ~Executors() = default;

  // TODO Use Executor index
  void emplace(const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index,
               std::unique_ptr<IExecutor> exec);

  IExecutor *at(const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index) const;

  uint32_t inputSize() const;

  uint32_t outputSize() const;

  const ir::OperandInfo inputInfo(const ir::IOIndex &index);

  const ir::OperandInfo outputInfo(const ir::IOIndex &index);

  void execute(const IODescription &desc);

private:
  void checkSupportedMultimodel() const;
  void executeModels(const IODescription &desc);
  uint16_t modelCount() const;

private:
  std::unordered_map<std::pair<ir::ModelIndex, ir::SubgraphIndex>, std::unique_ptr<IExecutor>>
    _executors;
  // NOTE _model_edges may use different struct type for executor implementation
  std::unique_ptr<ir::ModelEdges> _model_edges;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTORS_H__
