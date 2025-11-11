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

#include "BulkPipeLayer.h"

#include <iostream>
#include <memory>

namespace onert
{
namespace backend
{
namespace trix
{
namespace ops
{

BulkPipeLayer::BulkPipeLayer() : _inputs(), _outputs()
{
  // DO NOTHING
}

BulkPipeLayer::~BulkPipeLayer()
{
  // DO NOTHING - _pipeline_manager will be automatically cleaned up by unique_ptr
}

void BulkPipeLayer::configure(const std::vector<const IPortableTensor *> &inputs,
                              std::vector<IPortableTensor *> &outputs,
                              const std::vector<std::string> &binary_path)
{
  _inputs = inputs;
  _outputs = outputs;

  // Configure BulkPipeLineManager
  BulkPipelineManager::PipelineConfig config;
  config.model_paths = binary_path;
  config.device_id = 0;        // default device id = 0
  config.buffer_pool_size = 2; // Use 2 buffers

  _pipeline_manager = std::make_unique<BulkPipelineManager>(config);

  if (!_pipeline_manager->initialize())
  {
    throw std::runtime_error("Failed to initialize bulk pipeline manager");
  }
}

void BulkPipeLayer::run()
{
  if (!_pipeline_manager)
  {
    throw std::runtime_error("Pipeline manager is not initialized");
  }

  try
  {
    _pipeline_manager->execute(_inputs, _outputs);
  }
  catch (const std::exception &e)
  {
    std::cerr << "BulkPipeLayer execution failed: " << e.what() << std::endl;
    throw;
  }
}

void BulkPipeLayer::prepare()
{
  // DO NOTHING
}

} // namespace ops
} // namespace trix
} // namespace backend
} // namespace onert
