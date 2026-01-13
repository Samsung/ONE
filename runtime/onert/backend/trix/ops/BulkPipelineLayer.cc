/*
 * Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "BulkPipelineLayer.h"

#include <iostream>
#include <memory>

namespace onert::backend::trix::ops
{

BulkPipelineLayer::BulkPipelineLayer() : _inputs(), _outputs()
{
  // DO NOTHING
}

BulkPipelineLayer::~BulkPipelineLayer()
{
  // DO NOTHING - _pipeline_manager will be automatically cleaned up by unique_ptr
}

void BulkPipelineLayer::configure(const std::vector<const IPortableTensor *> &inputs,
                                  std::vector<IPortableTensor *> &outputs,
                                  const std::vector<std::string> &binary_path)
{
  _inputs = inputs;
  _outputs = outputs;

  // Configure BulkPipeLineManager
  BulkPipelineManager::PipelineConfig config;
  config.model_paths = binary_path;
  config.device_id = 0;      // default device id = 0
  config.n_owner_models = 2; // Use 2 owner models for buffer sharing
  config.n_inputs = inputs.size();
  config.n_outputs = outputs.size();

  _pipeline_manager = std::make_unique<BulkPipelineManager>(config);

  if (!_pipeline_manager->initialize())
  {
    throw std::runtime_error("Failed to initialize BulkPipelineManager");
  }
}

void BulkPipelineLayer::run()
{
  try
  {
    _pipeline_manager->execute(_inputs, _outputs);
  }
  catch (const std::exception &e)
  {
    _pipeline_manager->shutdown();
    std::cerr << "BulkPipelineLayer execution failed: " << e.what() << std::endl;
    throw;
  }
}

void BulkPipelineLayer::prepare()
{
  // DO NOTHING
}

} // namespace onert::backend::trix::ops
