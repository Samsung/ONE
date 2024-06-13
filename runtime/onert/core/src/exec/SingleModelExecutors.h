/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "ir/NNPkg.h"

namespace onert
{
namespace exec
{

/**
 * @brief Class to gather executor set for single model NN package
 */
class SingleModelExecutors : public IExecutors
{
public:
  /**
   * @brief Construct a new SingleModelExecutors object
   */
  SingleModelExecutors(void) = default;
  SingleModelExecutors(const SingleModelExecutors &) = delete;
  SingleModelExecutors(SingleModelExecutors &&) = default;

  /**
   * @brief Destroy the SingleModelExecutors object
   */
  ~SingleModelExecutors() = default;

public:
  void emplace(const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index,
               std::unique_ptr<IExecutor> exec) override;

  IExecutor *at(const ir::ModelIndex &model_index,
                const ir::SubgraphIndex &subg_index) const override;

  uint32_t inputSize() const override;

  uint32_t outputSize() const override;

  const ir::OperandInfo &inputInfo(const ir::IOIndex &index) const override;

  const ir::OperandInfo &outputInfo(const ir::IOIndex &index) const override;

  void execute(const ExecutionContext &ctx) override;

private:
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<IExecutor>> _executors;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_SINGLE_MODEL_EXECUTORS_H__
