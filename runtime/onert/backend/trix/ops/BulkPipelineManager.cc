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
