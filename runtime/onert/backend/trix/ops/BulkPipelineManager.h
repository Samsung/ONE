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

#ifndef __ONERT_BACKEND_TRIX_OPS_BULKPIPELINEMANAGER_H__
#define __ONERT_BACKEND_TRIX_OPS_BULKPIPELINEMANAGER_H__

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
    size_t buffer_pool_size{2};
  };

public:
  explicit BulkPipelineManager(const PipelineConfig &config);
  ~BulkPipelineManager();

  BulkPipelineManager(const BulkPipelineManager &) = delete;
  BulkPipelineManager &operator=(const BulkPipelineManager &) = delete;
  BulkPipelineManager(BulkPipelineManager &&) = delete;
  BulkPipelineManager &operator=(BulkPipelineManager &&) = delete;

  bool initialize();
  void shutdown();
  bool isInitialized() const { return _initialized; }

  void execute(const std::vector<const IPortableTensor *> &inputs,
               std::vector<IPortableTensor *> &outputs);

  size_t modelCount() const { return _models.size(); }
  std::shared_ptr<BulkPipelineModel> getModel(size_t index);

private:
  void setupPipeline();
  void createModels();
  void linkModels();
  void prepareModels();

  void handleError(const std::string &error);

private:
  PipelineConfig _config;
  std::atomic<bool> _initialized{false};

  std::vector<std::shared_ptr<BulkPipelineModel>> _models;

  std::array<std::shared_ptr<BulkPipelineModel>, 2> _buffer_pool;
  std::atomic<bool> _executing{false};
  mutable std::mutex _state_mutex;
  std::exception_ptr _last_error;
};

class BulkPipelineManagerException : public std::runtime_error
{
public:
  explicit BulkPipelineManagerException(const std::string &msg) : std::runtime_error(msg) {}
};

class PipelineInitializationException : public BulkPipelineManagerException
{
public:
  explicit PipelineInitializationException(const std::string &msg)
    : BulkPipelineManagerException(msg)
  {
  }
};

class PipelineExecutionException : public BulkPipelineManagerException
{
public:
  explicit PipelineExecutionException(const std::string &msg) : BulkPipelineManagerException(msg) {}
};

} // namespace ops
} // namespace trix
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRIX_OPS_BULKPIPELINEMANAGER_H__
