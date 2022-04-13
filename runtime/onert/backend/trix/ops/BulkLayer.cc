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

#include "BulkLayer.h"
#include <util/logging.h>

#include <libnpuhost.h>

namespace onert
{
namespace backend
{
namespace trix
{
namespace ops
{

BulkLayer::BulkLayer() : _inputs(), _outputs(), _model_id(0), _meta(nullptr), _dev_context(nullptr)
{
  // DO NOTHING
}

BulkLayer::~BulkLayer() = default;

void BulkLayer::configure(const std::vector<const IPortableTensor *> &inputs,
                          std::vector<IPortableTensor *> &outputs, std::string binary_path,
                          const std::shared_ptr<DevContext> &dev_context)
{
  _inputs = inputs;
  _outputs = outputs;
  _dev_context = dev_context;

  _meta = getNPUmodel_metadata(binary_path.c_str(), false);
  if (_meta == nullptr)
  {
    throw std::runtime_error("Unable to extract the model metadata");
  }

  generic_buffer model_file;
  model_file.type = BUFFER_FILE;
  model_file.filepath = binary_path.c_str();
  model_file.size = _meta->size;

  if (registerNPUmodel(dev_context->getDev(), &model_file, &_model_id) < 0)
  {
    throw std::runtime_error("Failed to register npu model");
  }
}

void BulkLayer::run()
{
  int req_id;
  if (createNPU_request(_dev_context->getDev(), _model_id, &req_id))
  {
    throw std::runtime_error("Unable to create NPU request with model id (" +
                             std::to_string(_model_id) + ")");
  }

  if (_meta->input_seg_num != _inputs.size())
  {
    throw std::runtime_error("input size does not match to model input seg num");
  }

  if (_meta->output_seg_num != _outputs.size())
  {
    throw std::runtime_error("output size does not match to model output seg num");
  }

  tensors_data_info in_info;
  tensors_data_info out_info;
  _dev_context->setDataInfo<const IPortableTensor>(&in_info, _inputs);
  _dev_context->setDataInfo<IPortableTensor>(&out_info, _outputs);

  input_buffers input_buf;
  output_buffers output_buf;
  _dev_context->setBuffer<const IPortableTensor>(&input_buf, _inputs);
  _dev_context->setBuffer<IPortableTensor>(&output_buf, _outputs);

  if (setNPU_requestData(_dev_context->getDev(), req_id, &input_buf, &in_info, &output_buf,
                         &out_info))
  {
    throw std::runtime_error("Unable to create NPU request for model id (" +
                             std::to_string(_model_id) + ")");
  }

  if (submitNPU_request(_dev_context->getDev(), req_id))
  {
    throw std::runtime_error("Unable to submit NPU request with req id (" + std::to_string(req_id) +
                             ")");
  }

  if (removeNPU_request(_dev_context->getDev(), req_id))
  {
    throw std::runtime_error("Unable to remove NPU request with req id (" + std::to_string(req_id) +
                             ")");
  }
}

void BulkLayer::prepare()
{
  // DO NOTHING
}

} // namespace ops
} // namespace trix
} // namespace backend
} // namespace onert
