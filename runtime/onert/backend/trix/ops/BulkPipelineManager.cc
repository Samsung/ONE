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

namespace onert::backend::trix::ops
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
    // Already initialized
    return true;
  }

  try
  {
    createModels();
    prepareModels();
    linkModels();

    _initialized = true;
    return true;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Failed to initialize pipeline: " + std::string(e.what()) << std::endl;
    shutdown();
    return false;
  }
}

void BulkPipelineManager::shutdown()
{
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
}

void BulkPipelineManager::execute(const std::vector<const IPortableTensor *> &inputs,
                                  std::vector<IPortableTensor *> &outputs)
{
  if (!_initialized.load())
  {
    throw std::runtime_error("Pipeline is not initialized");
  }

  if (_models.empty())
  {
    throw std::runtime_error("No models in pipeline");
  }

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
        throw std::runtime_error("Model at index " + std::to_string(i) + " is not prepared");
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
      if (_use_buffer_sharing)
      {
        if (auto next = model->getNextModel())
        {
          next->startAsyncBufferFill();
        }
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

void BulkPipelineManager::createModels()
{
  _models.clear();
  _models.reserve(_config.model_paths.size());

  for (size_t i = 0; i < _config.model_paths.size(); ++i)
  {
    auto model = std::make_shared<BulkPipelineModel>(_config.model_paths[i], _config.device_id);
    if (!model->initialize())
    {
      throw std::runtime_error("Failed to initialize model: " + model->modelPath());
    }
    _models.push_back(model);
  }

  auto first_values =
    std::pair<size_t, size_t>{_models.front()->programSize(), _models.front()->weightSize()};
  _use_buffer_sharing =
    std::all_of(_models.begin(), _models.end(), [first_values](const auto &model) {
      return model->programSize() == first_values.first &&
             model->weightSize() == first_values.second;
    });

  if (_use_buffer_sharing)
  {
    int model_idx = 0;
    for (auto model : _models)
    {
      if (model_idx++ < _config.n_owner_models)
      {
        // First n_shared_owner_models models are OWNERS
        continue;
      }

      // Other models are SHARED
      model->setBufferOwnership(BulkPipelineModel::BufferOwnership::SHARED);
    }
  }
}

void BulkPipelineManager::linkModels()
{
  // If models are not shared, no need to link them
  if (!_use_buffer_sharing)
    return;

  for (size_t i = 0; i < _models.size(); ++i)
  {
    if (i + _config.n_owner_models < _models.size())
    {
      _models[i]->setNextModel(_models[i + _config.n_owner_models]);
    }
    else
    {
      _models[i]->setNextModel(nullptr);
    }

    // Shared models share buffers from owners in buffer pool
    if (_models[i]->ownership() == BulkPipelineModel::BufferOwnership::SHARED)
    {
      size_t owner_index = i % _config.n_owner_models;
      _models[i]->shareBuffersFrom(*_models[owner_index]);
    }
  }
}

void BulkPipelineManager::prepareModels()
{
  for (auto &model : _models)
  {
    if (!model->prepare())
    {
      throw std::runtime_error("Failed to prepare model: " + model->modelPath());
    }
  }
}

} // namespace onert::backend::trix::ops
