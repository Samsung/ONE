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

  for (int i = 0; i < _dev_context->getDevSize(); i++)
  {
    if (registerNPUmodel(dev_context->getDev(i), &model_file, &_model_id[i]) < 0)
    {
      throw std::runtime_error("Failed to register npu model");
    }
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
  // We assume user wants batch execution if user's input size is multiples of model's input size
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

  std::vector<input_buffers> input_buf;
  std::vector<output_buffers> output_buf;
  input_buf.resize(_dev_context->getDevSize());
  output_buf.resize(_dev_context->getDevSize());

  std::vector<std::future<void>> f(_dev_context->getDevSize());

  const int num_cores = _dev_context->getDevSize();
  if (is_batch_execution)
  {
    // TODO: Support for general number of cores(>2)
    // Here we assume that 2 trix cores
    for (int i = 0; i < (batch_size); i = i + num_cores)
    {
      for (int core = 0; core < num_cores; core++)
      {
        _dev_context->setBuffer<const IPortableTensor>(&input_buf[core], _inputs, batch_size,
                                                       i + core);
        _dev_context->setBuffer<IPortableTensor>(&output_buf[core], _outputs, batch_size, i + core);
      }
      for (int core = 0; core < num_cores; core++)
      {

        if (i + core < batch_size)
        {
          f[core] =
            std::async(std::launch::async, &single_job, _dev_context->getDev(core), req_id[core],
                       &input_buf[core], &in_info, &output_buf[core], &out_info);
        }
      }
      for (int core = 0; core < num_cores; core++)
      {
        f[core].wait();
      }
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
