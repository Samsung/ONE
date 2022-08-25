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
#include <future>

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

BulkLayer::~BulkLayer() { free(_meta); }

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

  _model_id.resize(_dev_context->getDevSize());

  generic_buffer model_file;
  model_file.type = BUFFER_FILE;
  model_file.filepath = binary_path.c_str();
  model_file.size = _meta->size;

  if (registerNPUmodel(dev_context->getDev(0), &model_file, &_model_id[0]) < 0)
  {
    throw std::runtime_error("Failed to register npu model");
  }
  if (registerNPUmodel(dev_context->getDev(1), &model_file, &_model_id[1]) < 0)
  {
    throw std::runtime_error("Failed to register npu model");
  }
}

void single_job(npudev_h dev, int req_id, input_buffers *input_buf, tensors_data_info *in_info,
                output_buffers *output_buf, tensors_data_info *out_info)
{
  if (setNPU_requestData(dev, req_id, input_buf, in_info, output_buf, out_info))
  {
    throw std::runtime_error("Unable to create NPU request for red_id (" + std::to_string(req_id) +
                             ")");
  }

  if (submitNPU_request(dev, req_id))
  {
    throw std::runtime_error("Unable to submit NPU request with req id (" + std::to_string(req_id) +
                             ")");
  }
}

void BulkLayer::run()
{
  // TODO: Remove too many assumption
  // We assume user wants batch execurion if user's input size is larger than model's input size
  int user_input_batch = (_inputs[0]->get_info().shape()).dim(0);
  int model_input_batch = _meta->input_seg_dims[0][0];
  int batch_size = user_input_batch / model_input_batch;
  bool is_batch_execution = (batch_size != 1 ? true : false);

  std::vector<int> req_id(_dev_context->getDevSize());

  for (int i = 0; i < _dev_context->getDevSize(); i++)
  {
    if (createNPU_request(_dev_context->getDev(i), _model_id[i], &req_id[i]))
    {
      throw std::runtime_error("Unable to create NPU request with model id (" +
                               std::to_string(_model_id[i]) + ")");
    }
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

  input_buffers input_buf[2];
  output_buffers output_buf[2];

  if (is_batch_execution)
  {
    // TODO: Support for general number of cores(>2)
    // Here we assume that 2 trix cores
    for (int i = 0; i < (batch_size); i = i + 2)
    {
      _dev_context->setBuffer<const IPortableTensor>(&input_buf[0], _inputs, batch_size, i);
      _dev_context->setBuffer<IPortableTensor>(&output_buf[0], _outputs, batch_size, i);

      _dev_context->setBuffer<const IPortableTensor>(&input_buf[1], _inputs, batch_size, i + 1);
      _dev_context->setBuffer<IPortableTensor>(&output_buf[1], _outputs, batch_size, i + 1);

      auto f0 = std::async(std::launch::async, &single_job, _dev_context->getDev(0), req_id[0],
                           &input_buf[0], &in_info, &output_buf[0], &out_info);
      if (i + 1 < batch_size) // ignore last job if batch_size is odd number
      {
        auto f1 = std::async(std::launch::async, &single_job, _dev_context->getDev(1), req_id[1],
                             &input_buf[1], &in_info, &output_buf[1], &out_info);
        f1.wait();
      }
      f0.wait();
    }
  }
  else
  {
    _dev_context->setBuffer<const IPortableTensor>(&input_buf[0], _inputs, batch_size, 0);
    _dev_context->setBuffer<IPortableTensor>(&output_buf[0], _outputs, batch_size, 0);

    single_job(_dev_context->getDev(0), req_id[0], &input_buf[0], &in_info, &output_buf[0],
               &out_info);
  }

  for (int i = 0; i < _dev_context->getDevSize(); i++)
  {
    if (removeNPU_request(_dev_context->getDev(i), req_id[i]))
    {
      throw std::runtime_error("Unable to remove NPU request with req id (" +
                               std::to_string(req_id[i]) + ")");
    }
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
