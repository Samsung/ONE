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

#include "BulkPipelineManager.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <thread>
#include <chrono>

namespace onert
{
namespace backend
{
namespace trix
{
namespace ops
{

BulkPipelineManager::BulkPipelineManager(const PipelineConfig &config) : _config(config)
{
  // DO NOTHING
}

BulkPipelineManager::~BulkPipelineManager() { shutdown(); }

bool BulkPipelineManager::initialize()
{
  if (_initialized.load())
  {
    // Already initilized
    return true;
  }

  try
  {
    setupPipeline();
    createModels();
    prepareModels();
    linkModels();

    _initialized = true;
    return true;
  }
  catch (const std::exception &e)
  {
    handleError("Failed to initialize pipeline: " + std::string(e.what()));
    shutdown();
    return false;
  }
}

void BulkPipelineManager::shutdown()
{
  if (!_initialized.load())
  {
    return;
  }

  _initialized = false;

  // Wait until all executions are finished
  while (_executing.load())
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Release models and clear buffer pool
  for (auto &model : _models)
  {
    if (model)
    {
      model->release();
    }
  }
  _models.clear();
  // Clear buffer pool
  _buffer_pool.fill({});
}

void BulkPipelineManager::execute(const std::vector<const IPortableTensor *> &inputs,
                                  std::vector<IPortableTensor *> &outputs)
{
  if (!_initialized.load())
  {
    throw PipelineExecutionException("Pipeline is not initialized");
  }

  if (_models.empty())
  {
    throw PipelineExecutionException("No models in pipeline");
  }

  std::lock_guard<std::mutex> lock(_state_mutex);
  _executing = true;

  try
  {
    auto current_inputs = inputs;
    auto current_outputs = outputs;

    for (size_t i = 0; i < _models.size(); ++i)
    {
      auto &model = _models[i];
      if (!model || !model->isPrepared())
      {
        throw PipelineExecutionException("Model at index " + std::to_string(i) +
                                         " is not prepared");
      }

      // Wait for buffer ready before execution
      model->waitForBufferReady();

      // Execute model
      model->run(current_inputs, current_outputs);

      // The input of the next model is the output of the current model
      if (i < _models.size() - 1)
      {
        current_inputs.clear();
        for (const auto &output : current_outputs)
        {
          current_inputs.push_back(const_cast<IPortableTensor *>(output));
        }
      }

      // Prepare next shared neighbor model
      if (auto next = model->getNextModel())
      {
        next->startAsyncBufferFill();
      }
    }
  }
  catch (...)
  {
    _executing = false;
    throw;
  }

  _executing = false;
}

std::shared_ptr<BulkPipelineModel> BulkPipelineManager::getModel(size_t index)
{
  if (index >= _models.size())
  {
    return nullptr;
  }
  return _models[index];
}

void BulkPipelineManager::setupPipeline()
{
  if (_config.model_paths.empty())
  {
    throw PipelineInitializationException("No model paths provided in configuration");
  }

  if (_config.buffer_pool_size < 2)
  {
    throw PipelineInitializationException("Buffer pool size must be at least 2");
  }
}

void BulkPipelineManager::createModels()
{
  _models.clear();
  _models.reserve(_config.model_paths.size());

  for (size_t i = 0; i < _config.model_paths.size(); ++i)
  {
    // First `buffer_pool_size` models are owners of their own buffers, others share buffers from
    BulkPipelineModel::ModelOwnership ownership = (i < _config.buffer_pool_size)
                                                    ? BulkPipelineModel::ModelOwnership::OWNER
                                                    : BulkPipelineModel::ModelOwnership::SHARED;

    auto model =
      std::make_shared<BulkPipelineModel>(_config.model_paths[i], _config.device_id, ownership);
    _models.push_back(model);

    // Store owners in buffer pool for sharing purposes later
    if (i < _buffer_pool.size())
    {
      _buffer_pool[i] = model;
    }
  }
}

void BulkPipelineManager::linkModels()
{
  for (size_t i = 0; i < _models.size(); ++i)
  {
    if (i + _config.buffer_pool_size < _models.size())
    {
      _models[i]->setNextModel(_models[i + _buffer_pool.size()]);
    }
    else
    {
      _models[i]->setNextModel(nullptr);
    }

    // Shared models share buffers from owners in buffer pool
    if (_models[i]->ownership() == BulkPipelineModel::ModelOwnership::SHARED)
    {
      size_t owner_index = i % _buffer_pool.size();
      if (_buffer_pool[owner_index])
      {
        _models[i]->shareBuffersFrom(*_buffer_pool[owner_index]);
      }
    }
  }
}

void BulkPipelineManager::prepareModels()
{
  for (auto &model : _models)
  {
    if (!model->prepare())
    {
      throw PipelineInitializationException("Failed to prepare model: " + model->modelPath());
    }
  }
}

void BulkPipelineManager::handleError(const std::string &error)
{
  std::cerr << "BulkPipelineManager Error: " << error << std::endl;
  _last_error = std::current_exception();
}

} // namespace ops
} // namespace trix
} // namespace backend
} // namespace onert
