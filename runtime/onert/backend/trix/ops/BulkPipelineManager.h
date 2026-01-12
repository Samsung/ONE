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

#ifndef __ONERT_BACKEND_TRIX_OPS_BULK_PIPE_LINE_MANAGER_H__
#define __ONERT_BACKEND_TRIX_OPS_BULK_PIPE_LINE_MANAGER_H__

#include <memory>
#include <vector>
#include <string>
#include <array>
#include <atomic>
#include <mutex>
#include <exception>
#include <backend/IPortableTensor.h>
#include "BulkPipelineModel.h"

namespace onert
{
namespace backend
{
namespace trix
{
namespace ops
{

class BulkPipelineManager
{
public:
  struct PipelineConfig
  {
    std::vector<std::string> model_paths;
    int device_id{0};
  };

public:
  explicit BulkPipelineManager(const PipelineConfig &config);
  ~BulkPipelineManager();

  // Disallow copying
  BulkPipelineManager(const BulkPipelineManager &) = delete;
  BulkPipelineManager &operator=(const BulkPipelineManager &) = delete;

  bool initialize();
  void shutdown();
  bool isInitialized() const { return _initialized; }

  void execute(const std::vector<const IPortableTensor *> &inputs,
               std::vector<IPortableTensor *> &outputs);

private:
  void createModels();
  void prepareModels();

private:
  PipelineConfig _config;
  std::atomic<bool> _initialized{false};
  std::atomic<bool> _executing{false};

  std::vector<std::shared_ptr<BulkPipelineModel>> _models;
};

} // namespace ops
} // namespace trix
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRIX_OPS_BULK_PIPE_LINE_MANAGER_H__
