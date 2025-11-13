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

#ifndef __ONERT_BACKEND_TRIX_OPS_BULKPIPELINEMODEL_H__
#define __ONERT_BACKEND_TRIX_OPS_BULKPIPELINEMODEL_H__

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <backend/IPortableTensor.h>
#include <libnpuhost.h>

#include "BulkPipelineBuffer.h"

namespace onert
{
namespace backend
{
namespace trix
{
namespace ops
{

class BulkPipelineModel
{
public:
  enum class ModelOwnership
  {
    OWNER,
    SHARED
  };

public:
  BulkPipelineModel(const std::string &model_path, int device_id,
                    ModelOwnership ownership = ModelOwnership::OWNER);
  ~BulkPipelineModel();

  // Disallow copying allow moving
  BulkPipelineModel(const BulkPipelineModel &) = delete;
  BulkPipelineModel &operator=(const BulkPipelineModel &) = delete;
  BulkPipelineModel(BulkPipelineModel &&other) noexcept;
  BulkPipelineModel &operator=(BulkPipelineModel &&other) noexcept;

  bool prepare();
  void release();
  bool isPrepared() const { return _prepared; }

  void run(const std::vector<const IPortableTensor *> &inputs,
           std::vector<IPortableTensor *> &outputs);

  void shareBuffersFrom(const BulkPipelineModel &owner);
  void setNextModel(std::shared_ptr<BulkPipelineModel> next);
  std::shared_ptr<BulkPipelineModel> getNextModel() { return _next_model; };

  void waitForBufferReady();
  void markBufferReady();
  void startAsyncBufferFill();

  const npubin_meta *metadata() const { return _meta.get(); }
  uint32_t modelId() const { return _model_id; }
  npudev_h device() const { return _dev; }
  const std::string &modelPath() const { return _model_path; }
  ModelOwnership ownership() const { return _ownership; }

private:
  bool loadMetadata();
  void allocateBuffers();
  void fillBuffers();
  void registerModel();
  void unregisterModel();
  void openDevice();
  void closeDevice();

private:
  std::string _model_path;
  int _device_id;
  ModelOwnership _ownership;
  std::atomic<bool> _prepared{false};

  npudev_h _dev;
  uint32_t _model_id{0};

  std::unique_ptr<npubin_meta> _meta;
  size_t _meta_size{0};
  FILE *_fp{nullptr};

  std::shared_ptr<BulkPipelineBuffer> _program_buffer;
  std::shared_ptr<BulkPipelineBuffer> _weight_buffer;

  std::shared_ptr<BulkPipelineModel> _next_model;

  std::mutex _buffer_mutex;
  std::condition_variable _buffer_cv;
  std::atomic<bool> _buffer_ready{false};
  std::future<void> _async_fill_future;
};

class BulkPipelineModelException : public std::runtime_error
{
public:
  explicit BulkPipelineModelException(const std::string &msg) : std::runtime_error(msg) {}
};

class ModelPreparationException : public BulkPipelineModelException
{
public:
  explicit ModelPreparationException(const std::string &msg) : BulkPipelineModelException(msg) {}
};

class ModelExecutionException : public BulkPipelineModelException
{
public:
  explicit ModelExecutionException(const std::string &msg) : BulkPipelineModelException(msg) {}
};

} // namespace ops
} // namespace trix
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRIX_OPS_BULKPIPELINEMODEL_H__
